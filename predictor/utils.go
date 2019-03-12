package predictor

import (
	"image"
	"image/png"
	"os"
	"runtime"
	"runtime/debug"

	"github.com/pkg/errors"
	imagetypes "github.com/rai-project/image/types"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	gotensor "gorgonia.org/tensor"
)

func makeTensorFromGoTensor(in0 []*gotensor.Dense) (*tf.Tensor, error) {
	if len(in0) < 1 {
		return nil, errors.New("no dense tensor in input")
	}

	fst := in0[0]
	joined, err := fst.Concat(0, in0[1:]...)
	if err != nil {
		return nil, errors.Wrap(err, "unable to concat tensors")
	}
	joined.Reshape(append([]int{len(in0)}, fst.Shape()...)...)

	shape := make([]int64, len(joined.Shape()))
	for ii, s := range joined.Shape() {
		shape[ii] = int64(s)
	}

	switch t := in0[0].Dtype(); t {
	case gotensor.Uint8:
		return flattenedUint8ToTensor(joined.Data().([]uint8), shape)
	case gotensor.Uint16:
		return flattenedUint16ToTensor(joined.Data().([]uint16), shape)
	case gotensor.Uint32:
		return flattenedUint32ToTensor(joined.Data().([]uint32), shape)
	case gotensor.Int8:
		return flattenedInt8ToTensor(joined.Data().([]int8), shape)
	case gotensor.Int16:
		return flattenedInt16ToTensor(joined.Data().([]int16), shape)
	case gotensor.Int32:
		return flattenedInt32ToTensor(joined.Data().([]int32), shape)
	case gotensor.Float32:
		return flattenedFloat32ToTensor(joined.Data().([]float32), shape)
	case gotensor.Float64:
		return flattenedFloat64ToTensor(joined.Data().([]float64), shape)
	default:
		return nil, errors.Errorf("invalid element datatype %v", t)
	}
}

func reshapeTensorFloats(data [][]float32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]float32, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]float32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]float32, W)
			for w := int64(0); w < W; w++ {
				offset := C * (W*h + w)
				tw := ndata[offset : offset+C]
				th[w] = tw
			}
			tn[h] = th
		}
		tensor[n] = tn
	}
	return tf.NewTensor(tensor)
}

func reshapeTensorBytes(data [][]uint8, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint8, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]uint8, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint8, W)
			for w := int64(0); w < W; w++ {
				offset := C * (W*h + w)
				tw := ndata[offset : offset+C]
				th[w] = tw
			}
			tn[h] = th
		}
		tensor[n] = tn
	}
	return tf.NewTensor(tensor)
}

func decodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func makeTensorFromBytes(b []byte) (*tf.Tensor, error) {
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

func toPng(filePath string, imgByte []byte, bounds image.Rectangle) {

	img := imagetypes.NewRGBImage(bounds)
	copy(img.Pix, imgByte)

	out, _ := os.Create(filePath)
	defer out.Close()

	err := png.Encode(out, img.ToRGBAImage())
	if err != nil {
		log.Println(err)
	}
}

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

func forceGC() {
	runtime.GC()
	debug.FreeOSMemory()
}
