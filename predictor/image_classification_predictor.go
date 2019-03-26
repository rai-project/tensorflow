package predictor

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"os"
	"runtime"
	"strings"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/downloadmanager"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tensorflow"
	proto "github.com/rai-project/tensorflow"
	"github.com/rai-project/tracer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	gotensor "gorgonia.org/tensor"
)

type ImageClassificationPredictor struct {
	common.ImagePredictor
	tfGraph            *tf.Graph
	tfSession          *Session
	labels             []string
	inputLayer         string
	probabilitiesLayer string
	probabilities      interface{}
}

func NewImageClassificationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImageClassificationPredictor)

	return predictor.Load(context.Background(), model, opts...)
}

// Download ...
func (p *ImageClassificationPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &ImageClassificationPredictor{
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

func (p *ImageClassificationPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImageClassificationPredictor{
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

func (p *ImageClassificationPredictor) download(ctx context.Context) error {
	span, ctx := opentracing.StartSpanFromContext(
		ctx,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		return nil
	}
	checksum := p.GetGraphChecksum()
	if checksum == "" {
		return errors.New("Need graph file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download graph"),
	)

	if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetFeaturesChecksum()
	if checksum == "" {
		return errors.New("Need features file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download features"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	return nil
}

func (p *ImageClassificationPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	if p.tfSession != nil {
		return nil
	}

	span.LogFields(
		olog.String("event", "read features"),
	)

	var labels []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}
	p.labels = labels

	span.LogFields(
		olog.String("event", "read graph"),
	)
	model, err := ioutil.ReadFile(p.GetGraphPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return errors.Wrap(err, "unable to create tensorflow model graph")
	}

	modelReader := bytes.NewReader(model)
	p.inputLayer, err = p.GetInputLayerName(modelReader, "input_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get the input layer name")
	}
	p.probabilitiesLayer, err = p.GetOutputLayerName(modelReader, "probabilities_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get the probabilities layer name")
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

func (p ImageClassificationPredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
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

func (p ImageClassificationPredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
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

func (p *ImageClassificationPredictor) runOptions() *proto.RunOptions {
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		return &proto.RunOptions{
			TraceLevel: proto.RunOptions_SOFTWARE_TRACE,
		}
	}
	return nil
}

// Predict ...
func (p *ImageClassificationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "predict")
	defer span.Finish()

	if data == nil {
		return errors.New("input data nil")
	}
	input, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	session := p.tfSession
	graph := p.tfGraph

	tensor, err := makeTensorFromGoTensor(input)
	if err != nil {
		return err
	}

	sessionSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "c_predict")
	fetches, err := session.Run(ctx,
		map[tf.Output]*tf.Tensor{
			graph.Operation(p.inputLayer).Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation(p.probabilitiesLayer).Output(0),
		},
		nil,
		p.runOptions(),
	)
	sessionSpan.Finish()

	if err != nil {
		return errors.Wrapf(err, "failed to perform session.Run")
	}

	p.probabilities = fetches[0].Value()

	return nil
}

// ReadPredictedFeatures ...
func (p *ImageClassificationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	e, ok := p.probabilities.([][]float32)
	if !ok {
		return nil, errors.New("output is not of type [][]float32")
	}

	return p.CreateClassificationFeatures(ctx, e, p.labels)
}

func (p *ImageClassificationPredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *ImageClassificationPredictor) Close() error {
	if p.tfSession != nil {
		p.tfSession.Close()
	}
	forceGC()
	return nil
}

func (p ImageClassificationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageClassificationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorflow.FrameworkManifest
		agent.AddPredictor(framework, &ImageClassificationPredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
