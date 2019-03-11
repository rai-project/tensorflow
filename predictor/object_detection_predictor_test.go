package predictor

import (
	"context"
	"io/ioutil"
	"path/filepath"
	"testing"

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

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewObjectDetectionPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imagePath := filepath.Join(imgDir, "lane_control.jpg")
	b, err := ioutil.ReadFile(imagePath)
	if err != nil {
		panic(err)
  }
  image.Decode()
	input := make([][]byte, batchSize)
	input[0] = b

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

	pp.Println(pred[0][:3])
}
