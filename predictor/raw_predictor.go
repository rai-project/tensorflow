package predictor

import (
	"context"
	"io/ioutil"
	"runtime"
	"strings"

	"github.com/k0kubun/pp"
	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
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

type RawPredictor struct {
	common.RawPredictor
	tfGraph            *tf.Graph
	tfSession          *Session
	cu                 *cupti.CUPTI
	inputLayers        []string
	probabilitiesLayer string
	probabilities      interface{}
}

func NewRawPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()
	predictor := new(RawPredictor)
	return predictor.Load(ctx, model, opts...)
}

func (p *RawPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &RawPredictor{
		RawPredictor: common.RawPredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	ip.probabilitiesLayer, err = ip.GetOutputLayerName("probabilities_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the probabilities layer name")
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
func (p *RawPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &RawPredictor{
		RawPredictor: common.RawPredictor{
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

func (p *RawPredictor) download(ctx context.Context) error {
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

	return nil
}

func (p *RawPredictor) loadPredictor(ctx context.Context) error {
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
	if disableOptimizations {
		sessionConfig.GraphOptions = &tensorflow.GraphOptions{
			OptimizerOptions: &tensorflow.OptimizerOptions{
				DoFunctionInlining: false,
				GlobalJitLevel:     tensorflow.OptimizerOptions_OFF,
				OptLevel:           tensorflow.OptimizerOptions_L0,
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

// Predict ...
func (p *RawPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {

	inputTypes, err := p.GetInputParams("input_type")
	if err != nil {
		return err
	}
	p.inputLayers, err = p.GetInputParams("input_layer")
	if err != nil {
		return err
	}

	session := p.tfSession
	graph := p.tfGraph

	sessionSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict",
		opentracing.Tags{
			"evaluation_trace_level": p.TraceLevel(),
		})

	err = p.cuptiStart(ctx)
	if err != nil {
		return err
	}

	feeds := make(map[tf.Output]*Tensor, len(p.inputLayers))

	for ii, v := range p.inputLayers {
		input, err := p.GetInputDataByIdx(ii)
		if err != nil {
			return err
		}
		elementType, err := p.GetInputParamsByIdx(ii, "element_type")
		if err != nil {
			return err
		}

		var tensor *tf.Tensor
		switch inputTypes[ii] {
		case "scalar":
			switch elementType {
			case "int32":
				d := make([]int32, len(input))
				for ii, v := range input {
					d[ii] = v.(int32)
				}
				tensor, err = tf.NewTensor(d)
				if err != nil {
					return err
				}
			case "float32":
				d := make([]float32, len(input))
				for ii, v := range input {
					d[ii] = v.(float32)
				}
				pp.Println(len(d))
				tensor, err = tf.NewTensor(d)
				if err != nil {
					return err
				}
			default:
				return errors.Errorf("the scalar element type=%s is not valid", elementType)
			}
		case "slice":
			switch elementType {
			case "int32":
				d := make([][]int32, len(input))
				for ii, v := range input {
					d[ii] = v.([]int32)
				}
				tensor, err = tf.NewTensor(d)
				if err != nil {
					return err
				}
			case "float32":
				d := make([][]float32, len(input))
				for ii, v := range input {
					d[ii] = v.([]float32)
				}
				tensor, err = tf.NewTensor(d)
				if err != nil {
					return err
				}
			default:
				return errors.Errorf("the slice element type=%s is not valid", elementType)
			}

		default:
			return errors.New("input type not supported")
		}
		feeds[graph.Operation(v).Output(0)] = tensor
	}

	fetches, err := session.Run(ctx,
		feeds,
		[]tf.Output{
			graph.Operation(p.probabilitiesLayer).Output(0),
		},
		nil,
		p.runOptions(),
		p.GetGraphPath(),
	)

	p.cuptiClose()

	sessionSpan.Finish()

	if err != nil {
		return errors.Wrapf(err, "failed to perform session.Run")
	}

	p.probabilities = fetches[0].Value()
	// pp.Println(p.probabilities.([][]float32)[0])
	return nil
}

func (p *RawPredictor) cuptiStart(ctx context.Context) error {
	if p.TraceLevel() < tracer.SYSTEM_LIBRARY_TRACE {
		return nil
	}
	metrics := []string{}
	if p.RawPredictor.GPUMetrics() != "" {
		metrics = strings.Split(p.RawPredictor.GPUMetrics(), ",")
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

func (p RawPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.RawModality, nil
}

func (p *RawPredictor) runOptions() *proto.RunOptions {
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		return &proto.RunOptions{
			TraceLevel: proto.RunOptions_SOFTWARE_TRACE,
		}
	}
	return nil
}

func (p *RawPredictor) cuptiClose() {
	if p.cu == nil {
		return
	}
	p.cu.Wait()
	p.cu.Close()
	p.cu = nil
}

func (p *RawPredictor) Close() error {
	if p.tfSession != nil {
		p.tfSession.Close()
	}
	forceGC()
	return nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorflow.FrameworkManifest
		agent.AddPredictor(framework, &RawPredictor{
			RawPredictor: common.RawPredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
