package predictor

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/k0kubun/pp"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	tf "github.com/rai-project/tensorflow"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

func normalizeImageHWC(in *types.RGBImage, mean []float32, scale float32) ([]float32, error) {
	height := in.Bounds().Dy()
	width := in.Bounds().Dx()
	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := y*in.Stride + x*3
			rgb := in.Pix[offset : offset+3]
			r, g, b := rgb[0], rgb[1], rgb[2]
			out[offset+0] = (float32(r) - mean[0]) / scale
			out[offset+1] = (float32(g) - mean[1]) / scale
			out[offset+2] = (float32(b) - mean[2]) / scale
		}
	}
	return out, nil
}

func normalizeImageCHW(in *types.RGBImage, mean []float32, scale float32) ([]float32, error) {
	height := in.Bounds().Dy()
	width := in.Bounds().Dx()
	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := y*in.Stride + x*3
			rgb := in.Pix[offset : offset+3]
			r, g, b := rgb[0], rgb[1], rgb[2]
			out[y*width+x] = (float32(r) - mean[0]) / scale
			out[width*height+y*width+x] = (float32(g) - mean[1]) / scale
			out[2*width*height+y*width+x] = (float32(b) - mean[2]) / scale
		}
	}
	return out, nil
}
func TestPredictorNew(t *testing.T) {
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

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewImageClassificationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "platypus.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}
	img, err := image.Read(r)
	if err != nil {
		panic(err)
	}

	height := 227
	width := 227
	channels := 3
	resized, err := image.Resize(img, image.Resized(height, width))
	if err != nil {
		panic(err)
	}

	input := make([]*gotensor.Dense, batchSize)
	// imgBytes := resized.(*types.RGBImage).Pix
	imgFloats, err := normalizeImageHWC(resized.(*types.RGBImage), []float32{123, 117, 104}, 1.0)
	if err != nil {
		panic(err)
	}

	for ii := 0; ii < batchSize; ii++ {
		input[ii] = gotensor.New(
			gotensor.WithShape(height, width, channels),
			gotensor.WithBacking(imgFloats),
		)
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
