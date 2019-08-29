package predictor

import (
	"context"
	"strings"

	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/tracer"
)

// ImageCaptioningPredictor struct
type ImageCaptioningPredictor struct {
	*ImagePredictor
	CNNInputLayer         string
	CNNStateLayer         string
	seqSentenceInputLayer string
	seqStateInputLayer    string
	seqSoftmaxLayer       string
	seqStateOutputLayer   string
	captions              interface{}
}

// NewImageCaptioningPredictor is a constructor for ImageCaptioningPredictor.
func NewImageCaptioningPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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
