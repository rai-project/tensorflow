package main

// #cgo linux CFLAGS: -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -lcuda -lcudart -L/usr/local/cuda/lib64
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>
import "C"

import (
	"context"
	"fmt"
	"image"
	"path/filepath"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	tf "github.com/rai-project/tensorflow"
	"github.com/rai-project/tensorflow/predict"

	"github.com/rai-project/config"

	//_ "github.com/rai-project/tracer/all"

	_ "github.com/rai-project/tracer/jaeger"
)

var (
	batchSize    = 64
	graph_url    = "https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/bvlc_alexnet_1.0/frozen_model.pb"
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
			res[y*w+x] = float32(b>>8) - mean[0]
			res[w*h+y*w+x] = float32(g>>8) - mean[1]
			res[2*w*h+y*w+x] = float32(r>>8) - mean[2]
			// res[3*(y*w+x)] = float32(b>>8) - mean[0]
			// res[3*(y*w+x)+1] = float32(g>>8) - mean[1]
			// res[3*(y*w+x)+2] = float32(r>>8) - mean[2]
		}
	}

	return res, nil
}

func main() {
	tf.Register()
	model, err := tf.FrameworkManifest.FindModel("bvlc-alexnet:1.0")

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	} else {
		panic("no GPU")
	}

	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.InputNode("data", []uint32{3, 227, 227}),
		options.OutputNode("prob"),
		options.BatchSize(uint32(batchSize)))

	predictor, err := predict.New(*model, options.WithOptions(opts))
	defer predictor.Close()

	imgDir, _ := filepath.Abs("../_fixtures")
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

	C.cudaProfilerStart()

	preds, err := predictor.Predict(ctx, input)
	if err != nil {
		return
	}

	C.cudaProfilerStop()

	_ = preds

	pp.Println(preds[0][0])

}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
