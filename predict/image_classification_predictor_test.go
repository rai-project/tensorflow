package predictor

import (
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	tf "github.com/rai-project/tensorflow"
	"github.com/stretchr/testify/assert"
)

var (
	batchSize = 1
)

// convert go Image to 1-dim array
func cvtImageTo1DArray(src image.Image, mean []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	b := src.Bounds()
	h := b.Max.Y - b.Min.Y // image height
	w := b.Max.X - b.Min.X // image width

	res := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(x+b.Min.X, y+b.Min.Y).RGBA()
			res[3*(y*w+x)] = float32(b>>8) - mean[0]
			res[3*(y*w+x)+1] = float32(g>>8) - mean[1]
			res[3*(y*w+x)+2] = float32(r>>8) - mean[2]
		}
	}

	return res, nil
}

func XXXTestPredictorNew(t *testing.T) {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("bvlc-alexnet:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := NewImageClassificationPredictor(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)

	defer predictor.Close()

	imgPredictor, ok := predictor.(*ImageClassificationPredictor)
	assert.True(t, ok)

	assert.NotEmpty(t, imgPredictor.tfGraph)
	assert.NotEmpty(t, imgPredictor.tfSession)

}

func TestImageClassification(t *testing.T) {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("bvlc-alexnet:1.0")
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

	predictor, err := NewImageClassificationPredictor(*model, options.WithOptions(opts))
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

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
