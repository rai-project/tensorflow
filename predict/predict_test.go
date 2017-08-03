package predict

import (
	"os"
	"testing"

	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	tf "github.com/rai-project/tensorflow"
	"github.com/stretchr/testify/assert"
	context "golang.org/x/net/context"
)

func XXXTestPredictLoad(t *testing.T) {
	framework := tf.FrameworkManifest
	model, err := framework.FindModel("vgg19:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := New(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)

	defer predictor.Close()

	imgPredictor, ok := predictor.(*ImagePredictor)
	assert.True(t, ok)

	assert.NotEmpty(t, imgPredictor.imageDimensions)
	assert.NotEmpty(t, imgPredictor.meanImage)

}

func TestPredictInference(t *testing.T) {
	framework := tf.FrameworkManifest
	model, err := framework.FindModel("inception:3.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := New(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	ctx := context.Background()
	err = predictor.Download(ctx)
	assert.NoError(t, err)

	preds, err := predictor.Predict(ctx, "http://buzzsharer.com/wp-content/uploads/2015/06/beautiful-running-horse.jpg")
	assert.NoError(t, err)
	if err != nil {
		return
	}
	_ = preds
	if false {
		pp.Println(preds)
	}

	preds.Sort()

	pp.Println(preds.Take(3))

}

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
