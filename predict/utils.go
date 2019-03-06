package predictor

import (
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func zeros(height, width, channels int) [][][]float32 {
	rows := make([][][]float32, height)
	for ii := range rows {
		columns := make([][]float32, width)
		for jj := range columns {
			columns[jj] = make([]float32, channels)
		}
		rows[ii] = columns
	}
	return rows
}

// func  makeTensorFromData(data [][]float32) (*tf.Tensor, error) {
// 	imageDims, err := p.GetImageDimensions()
// 	if err != nil {
// 		return nil, err
// 	}

// 	channels, height, width := int64(imageDims[0]), int64(imageDims[1]), int64(imageDims[2])
// 	batchSize := int64(p.BatchSize())
// 	if batchSize == 0 {
// 		batchSize = 1
// 	}

// 	shapeLen := width * height * channels
// 	dataLen := int64(len(data))
// 	if batchSize > dataLen {
// 		padding := make([]float32, (batchSize-dataLen)*shapeLen)
// 		data = append(data, padding)
// 	}

// 	return NewTensor(ctx, data, []int64{batchSize, height, width, channels})
// }

func decodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		return nil, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, err := decodeJpegGraph()
	if err != nil {
		return nil, err
	}
	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}
