package main

import "C"

import (
	"context"
	"fmt"
	"image"
	"path/filepath"

	"github.com/Unknwon/com"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlframework/framework/options"
	tf "github.com/rai-project/tensorflow"
	"github.com/rai-project/tensorflow/predict"

	"github.com/rai-project/config"
	// _ "github.com/rai-project/tracer/jaeger"
)

var (
	batchSize    = 64
	graph_url    = "s3.amazonaws.com/store.carml.org/models/tensorflow/models/bvlc_alexnet_1.0/frozen_model.pb"
	features_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
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
			res[3*(y*w+x)+1] = float32(g>>8) - mean[1]
			res[3*(y*w+x)+2] = float32(r>>8) - mean[2]
		}
	}

	return res, nil
}

func main() {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("bvlc-alexnet:1.0")
	if err != nil {
		panic(err)
	}

	device := options.CPU_DEVICE
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.InputNode("data", []int{3, 227, 227}),
		options.OutputNode("prob"),
		options.BatchSize(batchSize))

	predictor, err := predict.New(*model, options.WithOptions(opts))
	defer predictor.Close()

	imgDir, _ := com.GetSrcPath("github.com/rai-project/tensorflow/predict/examples/image_classification/")
	imagePath := filepath.Join(imgDir, "octopus.jpg")
	img, err := imgio.Open(imagePath)
	if err != nil {
		panic(err)
	}

	resized := transform.Resize(img, 227, 227, transform.Linear)
	res, err := cvtImageTo1DArray(resized, []float32{123, 117, 104})
	if err != nil {
		panic(err)
	}
	input := make([][]float32, batchSize)
	for ii := 0; ii < batchSize; ii++ {
		input[ii] = res
	}

	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}

	preds, err := predictor.ReadPredictedFeatures(ctx)
	if err != nil {
		panic(err)
	}

	pp.Println(preds[0][0])

}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
