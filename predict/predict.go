package predict

import (
	"bufio"
	"io/ioutil"
	"os"
	"strings"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	"github.com/rai-project/tensorflow"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	context "golang.org/x/net/context"
)

type ImagePredictor struct {
	common.ImagePredictor
	features  []string
	tfGraph   *tf.Graph
	tfSession *tf.Session
}

func New(model dlframework.ModelManifest, opts dlframework.PredictionOptions) (common.Predictor, error) {
	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImagePredictor)

	return predictor.Load(context.Background(), model, opts)
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts dlframework.PredictionOptions) (common.Predictor, error) {
	span, newCtx := tracer.StartSpanFromContext(ctx, "Load")
	ctx = newCtx
	defer span.Finish()

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
				Framework:         framework,
				Model:             model,
				PredictionOptions: opts,
				Tracer:            tracer,
			},
			WorkDir: workDir,
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

func (p *ImagePredictor) GetPreprocessOptions(ctx context.Context) (common.PreprocessOptions, error) {
	mean, err := p.GetMeanImage()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	scale, err := p.GetScale()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return common.PreprocessOptions{}, err
	}

	return common.PreprocessOptions{
		MeanImage: mean,
		Scale:     scale,
		Size:      []int{int(imageDims[1]), int(imageDims[2])},
		ColorMode: types.RGBMode,
		Layout:    image.HWCLayout,
	}, nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := p.GetTracer().StartSpanFromContext(
		ctx,
		"Download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		return nil
	}
	checksum := p.GetGraphChecksum()
	if checksum == "" {
		return errors.New("Need graph file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download graph"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	checksum = p.GetFeaturesChecksum()
	if checksum == "" {
		return errors.New("Need features file checksum in the model manifest")
	}

	span.LogFields(
		olog.String("event", "download features"),
	)
	if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	span, ctx := p.GetTracer().StartSpanFromContext(ctx, "LoadPredictor")
	defer span.Finish()

	if p.tfSession != nil {
		return nil
	}

	span.LogFields(
		olog.String("event", "read features"),
	)

	var features []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		features = append(features, line)
	}
	p.features = features

	span.LogFields(
		olog.String("event", "read graph"),
	)
	model, err := ioutil.ReadFile(p.GetGraphPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
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

func zeros(height, width, channels int) [][][]float32 {
	rows := make([][][]float32, height)
	for ii := range rows {
		columns := make([][]float32, width)
		for jj := range columns {
			pixels := make([]float32, channels)
			columns[jj] = pixels
		}
		rows[ii] = columns
	}
	return rows
}

// Needs NHWC
func (p *ImagePredictor) makeTensorFromImageData(ctx context.Context, data0 [][]float32) (*tf.Tensor, error) {
	span, ctx := opentracing.StartSpanFromContext(ctx, "makeTensorFromImageData")
	defer span.Finish()

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return nil, err
	}
	channels, height, width := int(imageDims[0]), int(imageDims[1]), int(imageDims[2])
	batchSize := int(p.BatchSize())

	makeImage := func(arry []float32) [][][]float32 {
		rows := make([][][]float32, height)
		for ii := range rows {
			columns := make([][]float32, width)
			for jj := range columns {
				pixels := make([]float32, channels)
				for kk := range pixels {
					pixels[kk] = arry[channels*(width*ii+jj)+kk]
				}
				columns[jj] = pixels
			}
			rows[ii] = columns
		}
		return rows
	}

	data := make([][][][]float32, batchSize)
	for ii, e := range data0 {
		data[ii] = makeImage(e)
	}
	// perform padding
	if len(data0) < batchSize {
		z := zeros(height, width, channels)
		for ii := len(data0); ii < batchSize; ii++ {
			data[ii] = z
		}
	}

	tensor, err := tf.NewTensor(data)
	if err != nil {
		return nil, err
	}

	return tensor, nil
}

func (p *ImagePredictor) Predict(ctx context.Context, data [][]float32, opts dlframework.PredictionOptions) ([]dlframework.Features, error) {
	span, ctx := p.GetTracer().StartSpanFromContext(ctx, "Predict", opentracing.Tags{
		"model_name":        p.Model.GetName(),
		"model_version":     p.Model.GetVersion(),
		"framework_name":    p.Model.GetFramework().GetName(),
		"framework_version": p.Model.GetFramework().GetVersion(),
		"batch_size":        p.BatchSize(),
	})
	defer span.Finish()

	session := p.tfSession
	graph := p.tfGraph

	tensor, err := p.makeTensorFromImageData(ctx, data)
	if err != nil {
		return nil, errors.New("cannot make tensor from image data")
	}

	fetches, err := session.Run(
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
	// labels for each image in the "batch".
	probabilities := fetches[0].Value().([][]float32)

	// pp.Println("rank = ", len(probabilities))
	// pp.Println("len[0] = ", len(probabilities[0]))
	// pp.Println("len[0:10] = ", probabilities[0][0:10])

	batchSize := int(opts.GetBatchSize())
	if batchSize == 0 {
		batchSize = 1
	}

	var output []dlframework.Features
	for _, prob := range probabilities {
		rprobs := make([]*dlframework.Feature, len(prob))
		for j, v := range prob {
			rprobs[j] = &dlframework.Feature{
				Index:       int64(j),
				Name:        "<> " + p.features[j],
				Probability: v,
			}
		}
		output = append(output, rprobs)
	}

	return output, nil
}

func (p *ImagePredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *ImagePredictor) Close() error {
	if p.tfSession != nil {
		p.tfSession.Close()
	}
	return nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorflow.FrameworkManifest
		agent.AddPredictor(framework, &ImagePredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
