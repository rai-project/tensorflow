package predictor

import (
	"context"
	"io"
	"io/ioutil"
	"runtime"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/downloadmanager"
	cupti "github.com/rai-project/go-cupti"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tensorflow"
	proto "github.com/rai-project/tensorflow"
	"github.com/rai-project/tracer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type ImagePredictor struct {
	common.ImagePredictor
	tfGraph   *tf.Graph
	tfSession *Session
}

func (p *ImagePredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
	model := p.Model
	modelInputs := model.GetInputs()
	typeParameters := modelInputs[0].GetParameters()
	name, err := p.GetTypeParameter(typeParameters, layer)
	if err != nil {
		graphDef, err := tensorflow.FromCheckpoint(reader)
		if err != nil {
			return "", errors.Wrap(err, "failed to read metagraph from checkpoint")
		}
		nodes := graphDef.GetNode()
		if nodes == nil {
			return "", errors.New("failed to read graph nodes")
		}
		// get the first node which has no input
		for _, n := range nodes {
			if len(n.GetInput()) == 0 {
				return n.GetName(), nil
			}
		}
		return "", errors.New("cannot determin the name of the input layer")
	}
	return name, nil
}

func (p *ImagePredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
	model := p.Model
	modelOutput := model.GetOutput()
	typeParameters := modelOutput.GetParameters()
	name, err := p.GetTypeParameter(typeParameters, layer)
	if err != nil {
		graphDef, err := tensorflow.FromCheckpoint(reader)
		if err != nil {
			return "", errors.Wrap(err, "failed to read metagraph from checkpoint")
		}
		nodes := graphDef.GetNode()
		if nodes == nil {
			return "", errors.New("failed to read graph nodes")
		}
		if len(nodes) == 0 {
			return "", errors.New("cannot determin the name of the output layer")
		}
		// get the last node in the graph
		return nodes[len(nodes)-1].GetName(), nil
	}
	return name, nil
}

func (p *ImagePredictor) Close() error {
	if p.tfSession != nil {
		p.tfSession.Close()
	}
	forceGC()
	return nil
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (*ImagePredictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

// Download ...
func (p *ImagePredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := opentracing.StartSpanFromContext(
		ctx,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
		},
	)
	defer span.Finish()

	model := p.Model

	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		if _, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx)); err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
	} else {
		span.LogFields(
			olog.String("event", "download model graph"),
		)
		checksum := p.GetGraphChecksum()
		if checksum != "" {
			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
				return err
			}
		}
	}

	if p.GetFeaturesUrl() != "" {
		span.LogFields(
			olog.String("event", "download features"),
		)
		checksum := p.GetFeaturesChecksum()
		if checksum != "" {
			if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
				return err
			}
		}
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	if p.tfSession != nil {
		return nil
	}

	span.LogFields(
		olog.String("event", "read graph"),
	)

	graphPath := p.GetGraphPath()
	if graphPath == "" {
		return errors.New("graph path is empty")
	}
	model, err := ioutil.ReadFile(graphPath)
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", graphPath)
	}
	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return errors.Wrap(err, "unable to create tensorflow model graph")
	}

	// Create a session for inference over graph.
	var sessionConfig tensorflow.ConfigProto
	if p.Options.UsesGPU() {
		sessionConfig = tensorflow.ConfigProto{
			DeviceCount: map[string]int32{
				"GPU": int32(nvidiasmi.GPUCount),
			},
			// LogDevicePlacement: true,
			GpuOptions: &proto.GPUOptions{
				ForceGpuCompatible: true,
				// PerProcessGpuMemoryFraction: 0.5,
			},
		}
	} else {
		sessionConfig = tensorflow.ConfigProto{
			DeviceCount: map[string]int32{
				"CPU": int32(runtime.NumCPU()),
				"GPU": int32(0),
			},
			// LogDevicePlacement: true,
			GpuOptions: &tensorflow.GPUOptions{
				ForceGpuCompatible: false,
			},
		}
	}
	sessionOpts := &SessionOptions{}
	if buf, err := sessionConfig.Marshal(); err == nil {
		sessionOpts.Config = buf
	}
	session, err := NewSession(graph, sessionOpts)
	if err != nil {
		return errors.Wrap(err, "unable to create tensorflow session")
	}

	p.tfGraph = graph
	p.tfSession = session

	return nil
}

func (p *ImagePredictor) runOptions() *proto.RunOptions {
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		return &proto.RunOptions{
			TraceLevel: proto.RunOptions_SOFTWARE_TRACE,
		}
	}
	return nil
}

func (p *ImagePredictor) cuptiStart(ctx context.Context) (*cupti.CUPTI, error) {
	if !p.UseGPU() || p.TraceLevel() < tracer.HARDWARE_TRACE {
		return nil, nil
	}
	cu, err := cupti.New(cupti.Context(ctx), cupti.SamplingPeriod(0))
	if err != nil {
		return nil, err
	}
	return cu, nil
}

func (p *ImagePredictor) cuptiClose(cu *cupti.CUPTI) {
	if cu == nil {
		return
	}
	cu.Wait()
	cu.Close()
}
