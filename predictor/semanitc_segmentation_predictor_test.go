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

func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

func TestSemanticSegmentationInference(t *testing.T) {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("deeplabv3_mobilenetv2_pascal_voc:1.0")
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

	predictor, err := NewSemanticSegmentationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "lane_control.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}
	img, err := image.Read(r)
	if err != nil {
		panic(err)
	}

	height := img.Bounds().Dy()
	width := img.Bounds().Dx()
	channels := 3
	inputSize := 513
	resizeRatio := float32(inputSize) / float32(max(width, height))
	targetWidth := int(resizeRatio * float32(width))
	targetHeight := int(resizeRatio * float32(height))
	resized, err := image.Resize(img, image.Resized(targetHeight, targetWidth))
	if err != nil {
		panic(err)
	}
	input := make([]*gotensor.Dense, batchSize)
	imgBytes := resized.(*types.RGBImage).Pix
	for ii := 0; ii < batchSize; ii++ {
		input[ii] = gotensor.New(
			gotensor.WithShape(targetHeight, targetWidth, channels),
			gotensor.WithBacking(imgBytes),
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

	pp.Println(pred[0][:1])
}
