package tensorflow

// import (
// 	"io/ioutil"
// 	"os"
// 	"path/filepath"
// 	"testing"

// 	"github.com/k0kubun/pp"
// 	"github.com/rai-project/config"
// 	"github.com/rai-project/downloadmanager"
// 	"github.com/stretchr/testify/assert"
// 	tf "github.com/tensorflow/tensorflow/tensorflow/go"
// )

// func TestCheckpointReader(t *testing.T) {
// 	url := "http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz"
// 	targetDir := filepath.Join(config.App.TempDir, "testing", "inception")
// 	path, err := downloadmanager.Download(url, targetDir)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	assert.NotEmpty(t, path)
// 	path = filepath.Dir(path)
// 	pp.Println(path)

// 	metaPath := filepath.Join(path, "model.ckpt-157585")
// 	f, err := os.Open(metaPath)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}
// 	defer f.Close()

// 	rd, err := ioutil.ReadAll(f)
// 	if err != nil {
// 		return
// 	}

// 	if true {
// 		model, err := tf.LoadSavedModel("/tmp/carml/testing/mobilenet/mobilenet_v1_0.25_128.ckpt", []string{"checkpoint"}, nil)
// 		assert.NoError(t, err)
// 		_ = model
// 		return
// 	}

// 	if false {
// 		graph := tf.NewGraph()
// 		err = graph.Import(rd, "")
// 		assert.NoError(t, err)
// 		if err != nil {
// 			return
// 		}
// 		return
// 	}

// 	// if false {
// 	// s := op.NewScope()
// 	// tf.RestoreV2(s, op.Const(s, targetDir))
// 	// }

// 	info := &MetaGraphDef{}
// 	err = info.Unmarshal(rd)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pp.Println(info.GetMetaInfoDef())
// 	pp.Println(info.GetAssetFileDef())
// }

// func TestMain(m *testing.M) {
// 	config.Init(
// 		config.AppName("carml"),
// 		config.DebugMode(true),
// 		config.VerboseMode(true),
// 	)
// 	os.Exit(m.Run())
// }
