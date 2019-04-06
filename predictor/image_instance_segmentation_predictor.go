package predictor

import (
	"bytes"
	"context"
	"io/ioutil"
	"strings"

	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/tensorflow"
	"github.com/rai-project/tracer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	gotensor "gorgonia.org/tensor"
)

type InstanceSegmentationPredictor struct {
	*ImagePredictor
	inputLayer         string
	boxesLayer         string
	probabilitiesLayer string
	classesLayer       string
	masksLayer         string
	boxes              interface{}
	probabilities      interface{}
	classes            interface{}
	masks              interface{}
}

func NewInstanceSegmentationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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

	predictor := new(InstanceSegmentationPredictor)

	return predictor.Load(ctx, model, opts...)
}

func (self *InstanceSegmentationPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}

	p := &InstanceSegmentationPredictor{
		ImagePredictor: pred,
	}

	model, err := ioutil.ReadFile(p.GetGraphPath())
	if err != nil {
		return nil, errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
	}
	modelReader := bytes.NewReader(model)

	p.inputLayer, err = p.GetInputLayerName(modelReader, "input_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get input layer name")
	}
	p.boxesLayer, err = p.GetOutputLayerName(modelReader, "boxes_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the boxes layer name")
	}
	p.probabilitiesLayer, err = p.GetOutputLayerName(modelReader, "probabilities_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the probabilities layer name")
	}
	p.classesLayer, err = p.GetOutputLayerName(modelReader, "classes_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the classes layer name")
	}
	p.masksLayer, err = p.GetOutputLayerName(modelReader, "masks_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the masks layer name")
	}

	return p, nil
}

// Predict ...
func (p *InstanceSegmentationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
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

	tensor, err := makeTensorFromGoTensors(input)
	if err != nil {
		return err
	}

	sessionSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")

	cu, err := p.cuptiStart(ctx)
	if err != nil {
		return err
	}

	fetches, err := session.Run(ctx,
		map[tf.Output]*tf.Tensor{
			graph.Operation(p.inputLayer).Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation(p.boxesLayer).Output(0),
			graph.Operation(p.probabilitiesLayer).Output(0),
			graph.Operation(p.classesLayer).Output(0),
			graph.Operation(p.masksLayer).Output(0),
		},
		nil,
		p.runOptions(),
	)

	p.cuptiClose(cu)

	sessionSpan.Finish()

	if err != nil {
		return errors.Wrapf(err, "failed to perform session.Run")
	}

	p.boxes = fetches[0].Value()
	p.probabilities = fetches[1].Value()
	p.classes = fetches[2].Value()
	p.masks = fetches[3].Value()

	return nil
}

// ReadPredictedFeatures ...
func (p *InstanceSegmentationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	boxes, ok := p.boxes.([][][]float32)
	if !ok {
		return nil, errors.New("boxes is not of type [][][]float32")
	}
	probabilities, ok := p.probabilities.([][]float32)
	if !ok {
		return nil, errors.New("probabilities is not of type [][]float32")
	}
	classes, ok := p.classes.([][]float32)
	if !ok {
		return nil, errors.New("classes is not of type [][]float32")
	}
	masks, ok := p.masks.([][][][]float32)
	if !ok {
		return nil, errors.New("masks is not of type [][][][]float32")
	}

	labels, err := p.GetLabels()
	if err != nil {
		return nil, errors.New("cannot get the labels")
	}

	return p.CreateInstanceSegmentFeatures(ctx, probabilities, classes, boxes, masks, labels)
}

func (p InstanceSegmentationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageInstanceSegmentationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorflow.FrameworkManifest
		agent.AddPredictor(framework, &InstanceSegmentationPredictor{
			ImagePredictor: &ImagePredictor{
				ImagePredictor: common.ImagePredictor{
					Base: common.Base{
						Framework: framework,
					},
				},
			},
		})
	})
}
