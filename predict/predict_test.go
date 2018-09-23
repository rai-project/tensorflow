package predict

import (
	"os"
	"path/filepath"
	"testing"

	context "context"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/go-mxnet-predictor/utils"
	tf "github.com/rai-project/tensorflow"
	"github.com/stretchr/testify/assert"
)

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
	opts := options.New(options.Context(ctx))

	predictor, err := New(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	// load test image for predction
	img, err := imgio.Open(filepath.Join(sourcepath.MustAbsoluteDir(), "..", "scratch/data", "beautiful-running-horse.jpg"))
	if err != nil {
		panic(err)
	}
	// preprocess
	resized := transform.Resize(img, 224, 224, transform.Linear)
	res, err := utils.CvtImageTo2DArray(resized)
	if err != nil {
		panic(err)
	}

	preds, err := predictor.Predict(ctx, res)
	assert.NoError(t, err)
	if err != nil {
		return
	}
	_ = preds
	if false {
		pp.Println(preds)
	}

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
