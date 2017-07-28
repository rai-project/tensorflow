package predict

import (
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/k0kubun/pp"
	"github.com/pkg/errors"

	"github.com/rai-project/dlframework"
	common "github.com/rai-project/dlframework/frameworks/common/predict"
	"github.com/rai-project/downloadmanager"
	"github.com/rai-project/utils"

	"github.com/Unknwon/com"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type ImagePredictor struct {
	common.ImagePredictor
	meanImage       []float32
	imageDimensions []int32
	tfGraph         *tf.Graph
	tfSession       *tf.Session
	workDir         string
	graphFilePath   string
}

func New(model dlframework.ModelManifest) (common.Predictor, error) {
	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}
	return newImagePredictor(model)
}

func newImagePredictor(model dlframework.ModelManifest) (*ImagePredictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
			},
		},
		workDir: workDir,
	}

	if err := ip.setImageDimensions(); err != nil {
		return nil, err
	}

	if err := ip.setMeanImage(); err != nil {
		return nil, err
	}

	return ip, nil
}

func (p *ImagePredictor) makeSession() error {
	model, err := ioutil.ReadFile(p.graphFilePath)
	if err != nil {
		return errors.Wrapf(err, "unable to read graph file %v", p.graphFilePath)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return errors.Wrap(err, "unable to create tensorflow model graph")
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return errors.Wrap(err, "unable to create tensorflow session")
	}

	p.tfGraph = graph
	p.tfSession = session

	return nil
}

func (p *ImagePredictor) Download() error {
	model := p.Model
	var downloadPath string
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		_, err := downloadmanager.Download(baseURL, p.workDir)
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		downloadPath = p.workDir
	} else {
		var err error
		url := path.Join(model.Model.BaseUrl, model.Model.GetGraphPath()) // this is a url, so path is correct
		downloadPath, err = downloadmanager.Download(url, filepath.Join(p.workDir, model.Model.GetGraphPath()))
		if err != nil {
			return errors.Wrapf(err, "failed to download model graph from %v", url)
		}
	}
	pth := filepath.Join(downloadPath, model.Model.GetGraphPath())
	if !com.IsFile(pth) {
		return errors.Errorf("the graph file %v was not found or is not a file", pth)
	}
	p.graphFilePath = pth
	return nil
}

func (p *ImagePredictor) Preprocess(data interface{}) (interface{}, error) {

	var reader io.ReadCloser
	defer func() {
		if reader != nil {
			reader.Close()
		}
	}()

	if str, ok := data.(string); ok {
		if com.IsFile(str) {
			f, err := os.Open(str)
			if err != nil {
				return nil, errors.Wrapf(err, "unable to open file from %v", str)
			}
			reader = f
		} else if utils.IsURL(str) {
			_, err := url.Parse(str)
			if err != nil {
				return nil, errors.Wrapf(err, "unable to parse url %v", str)
			}
			pth, err := downloadmanager.Download(str, p.workDir)
			if err != nil {
				return nil, errors.Wrapf(err, "unable to download url %v", str)
			}
			if !com.IsFile(pth) {
				return nil, errors.Wrapf(err, "downloaded file %v not found", pth)
			}
			f, err := os.Open(pth)
			if err != nil {
				return nil, errors.Wrapf(err, "unable to open downloaded file %v", pth)
			}
			reader = f
		}
	} else if rdr, ok := data.(io.Reader); ok {
		reader = ioutil.NopCloser(rdr)
	} else {
		return nil, errors.New("unexpected input")
	}

	tensor, err := p.makeTensorFromImage(reader)
	if err != nil {
		return nil, err
	}

	return tensor, nil
}

func (p *ImagePredictor) Predict(data interface{}) (*dlframework.PredictionFeatures, error) {

	if p.tfSession == nil {
		if err := p.makeSession(); err != nil {
			return nil, err
		}
	}

	session := p.tfSession
	graph := p.tfGraph

	tensor, ok := data.(*tf.Tensor)
	if !ok {
		return nil, errors.New("expecting a *tf.Tensor input to predict")
	}

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to perform inference")
	}
	// output[0].Value() is a vector containing probabilities of
	// labels for each image in the "batch". The batch size was 1.
	// Find the most probably label index.
	probabilities := output[0].Value().([][]float32)[0]
	// pp.Println("probabilities == ", probabilities)
	rprobs := make([]*dlframework.PredictionFeature, len(probabilities))
	for ii, prob := range probabilities {
		rprobs[ii] = &dlframework.PredictionFeature{
			Index:       int64(ii),
			Probability: prob,
		}
	}

	res := dlframework.PredictionFeatures(rprobs)
	return &res, nil
}

func (p *ImagePredictor) Close() error {
	if p.tfSession != nil {
		p.tfSession.Close()
	}
	return nil
}

func dummy() {
	if false {
		pp.Println("....")
	}
}
