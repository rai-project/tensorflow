package predictor

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"runtime"
	"strconv"
	"strings"

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
	cu        *cupti.CUPTI
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

// GetMultipleInputLayerNames generalizes GetInputLayerName when multiple inputs are present.
func (p *ImagePredictor) GetMultipleInputLayerNames(reader io.Reader, layers []string) ([]string, error) {
	model := p.Model
	names := make([]string, len(layers))
	modelInputs := model.GetInputs()
	if len(modelInputs) != len(layers) {
		return names, errors.New("mismatch between number of input layers specified in yaml and number of input layers qetting")
	}

	for idx, modelInput := range modelInputs {
		typeParameter := modelInput.GetParameters()
		name, err := p.GetTypeParameter(typeParameter, layers[idx])

		if err != nil {
			fmt.Println("Error happened in index "+strconv.Itoa(idx)+": ", err)
			graphDef, err := tensorflow.FromCheckpoint(reader)
			if err != nil {
				return make([]string, len(layers)), errors.Wrap(err, "failed to read metagraph from checkpoint")
			}
			nodes := graphDef.GetNode()
			if nodes == nil {
				return make([]string, len(layers)), errors.New("failed to read graph nodes")
			}
		}
		names[idx] = name
	}

	return names, nil
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

	if ip.Options.DisableFrameworkAutoTuning() {
		disableFrameworkAutoTuning()
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
	span, ctx := tracer.StartSpanFromContext(ctx,
		tracer.APPLICATION_TRACE,
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

		_, _, err := downloadmanager.DownloadFile(
			p.GetGraphUrl(),
			p.GetGraphPath(),
			downloadmanager.MD5Sum(p.GetGraphChecksum()),
		)
		if err != nil {
			return errors.Wrapf(err, "failed to download model graph from %v", p.GetGraphUrl())
		}
	}

	if p.GetFeaturesUrl() != "" {
		span.LogFields(
			olog.String("event", "download features"),
		)
		_, _, err := downloadmanager.DownloadFile(
			p.GetFeaturesUrl(),
			p.GetFeaturesPath(),
			downloadmanager.MD5Sum(p.GetFeaturesChecksum()),
		)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if ctx != nil {
		span, _ := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
		defer span.Finish()
	}

	if p.tfSession != nil {
		return nil
	}

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

func (p *ImagePredictor) cuptiStart(ctx context.Context) error {
	if p.TraceLevel() < tracer.SYSTEM_LIBRARY_TRACE {
		return nil
	}
	metrics := []string{}
	if p.ImagePredictor.GPUMetrics() != "" {
		metrics = strings.Split(p.ImagePredictor.GPUMetrics(), ",")
	}

	cu, err := cupti.New(cupti.Context(ctx),
		cupti.SamplingPeriod(0),
		cupti.Metrics(metrics),
	)
	if err != nil {
		return err
	}

	p.cu = cu
	return nil
}

func (p *ImagePredictor) cuptiClose() {
	if p.cu == nil {
		return
	}
	p.cu.Wait()
	p.cu.Close()
	p.cu = nil
}
