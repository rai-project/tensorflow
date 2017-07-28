package predict

import (
	"bytes"
	"image"
	"io"
	"io/ioutil"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	"github.com/k0kubun/pp"
	"github.com/pkg/errors"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func (p *ImagePredictor) setImageDimensions() error {
	dims, err := p.GetImageDimensions()
	if err != nil {
		return err
	}
	p.imageDimensions = dims
	return nil
}

func (p *ImagePredictor) setMeanImage() error {
	mean, err := p.GetMeanImage()
	if err != nil {
		return err
	}
	p.meanImage = mean
	return nil
}

// Convert the image in reader to a Tensor suitable as input to the Inception model.
func (p *ImagePredictor) makeTensorFromImage(reader io.Reader) (*tf.Tensor, error) {
	bts, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(bts))
	if err != nil {
		return nil, err
	}
	// Construct a graph to normalize the image
	graph, input, output, err := p.constructGraphToNormalizeImage(bts)
	if err != nil {
		return nil, err
	}
	// Execute that graph to normalize this one image
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

// The model takes as input the image described by a Tensor in a very
// specific normalized format (a particular image size, shape of the input tensor,
// normalized pixel values etc.).
//
// This function constructs a graph of TensorFlow operations which takes as
// input a JPEG-encoded string and returns a tensor suitable as input to the
// inception model.
func (p *ImagePredictor) constructGraphToNormalizeImage(img []byte) (graph *tf.Graph, input, output tf.Output, err error) {
	// Some constants specific to the pre-trained model at
	// - The model was trained after with images scaled to pixels.
	// - The colors, represented as R, G, B in 1-byte each were converted to
	//   float using (value - Mean)/Scale.

	var width, height int32

	if len(p.imageDimensions) == 4 {
		width, height = p.imageDimensions[2], p.imageDimensions[3]
	} else if len(p.imageDimensions) == 2 {
		width, height = p.imageDimensions[0], p.imageDimensions[1]
	} else {
		err = errors.Errorf("invalid image dimensions %#v", p.imageDimensions)
		return
	}

	mean := p.meanImage
	scale := float32(1)

	// - input is a String-Tensor, where the string the JPEG-encoded image.
	// - The inception model takes a 4D tensor of shape
	//   [BatchSize, Height, Width, Colors=3], where each pixel is
	//   represented as a triplet of floats
	// - Apply normalization on each pixel and use ExpandDims to make
	//   this single image be a "batch" of size 1 for ResizeBilinear.
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)

	var decoder tf.Output
	chans, imageFormat, err := findImageFormat(img)
	if err != nil {
		err = errors.Wrapf(err, "unable to get metadata for input image")
		return
	}

	switch imageFormat {
	case "jpeg":
		decoder = op.DecodeJpeg(s, input, op.DecodeJpegChannels(chans))
	case "png":
		decoder = op.DecodePng(s, input, op.DecodePngChannels(chans))
	default:
		err = errors.Errorf("%v is not a supported image format", imageFormat)
		return
	}

	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s, decoder, tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{height, width})),
			op.Const(s.SubScope("mean"), mean)),
		op.Const(s.SubScope("scale"), scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func findImageFormat(img []byte) (int64, string, error) {
	r := bytes.NewBuffer(img)
	_, name, err := image.DecodeConfig(r)
	if err != nil {
		return 0, "", err
	}
	channels := int64(3) // todo implement me
	return channels, name, err
}

func dummy2() {
	if false {
		pp.Println("....")
	}
}
