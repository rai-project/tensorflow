package predict

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"

	context "context"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework/framework/options"
	tf "github.com/rai-project/tensorflow"
	"github.com/stretchr/testify/assert"
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
			res[y*w+x] = float32(b>>8) - mean[0]
			res[w*h+y*w+x] = float32(g>>8) - mean[1]
			res[2*w*h+y*w+x] = float32(r>>8) - mean[2]
		}
	}

	return res, nil
}

var (
	graph_url    = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/bvlc_alexnet_1.0/frozen_model.pb"
	features_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
)

func XXXTestPredictLoad(t *testing.T) {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("inception:3.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := New(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)

	defer predictor.Close()

	imgPredictor, ok := predictor.(*ImagePredictor)
	assert.True(t, ok)

	assert.NotEmpty(t, imgPredictor.tfGraph)
	assert.NotEmpty(t, imgPredictor.tfSession)

}

func TestPredictInference(t *testing.T) {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("inception:3.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.InputNode("data", []uint32{3, 227, 227}),
		options.OutputNode("prob"),
		options.BatchSize(uint32(batchSize)))

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

	var input [][]float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, 227, 227, transform.Linear)
		res, err := cvtImageTo1DArray(resized, []float32{123, 117, 104})
		if err != nil {
			panic(err)
		}
		input = append(input, res)
	}

	preds, err := predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	_ = preds

	pp.Println(preds[0])

}

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
