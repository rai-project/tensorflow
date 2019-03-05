package predictor

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	tf "github.com/rai-project/tensorflow"
	"github.com/stretchr/testify/assert"
)

func TestObjectDetectionInference(t *testing.T) {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("ssd_mobilenet_v1_coco:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.InputNode("data", []int{3, 227, 227}),
		options.OutputNode("prob"),
		options.BatchSize(batchSize))

	predictor, err := New(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imagePath := filepath.Join(imgDir, "platypus.jpg")
	img, err := imgio.Open(imagePath)
	if err != nil {
		panic(err)
	}

	input := make([][]float32, batchSize)
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, 227, 227, transform.Linear)
		res, err := cvtImageTo1DArray(resized, []float32{123, 117, 104})
		if err != nil {
			panic(err)
		}
		input[ii] = res
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pp.Println(pred[0][0])

}
