package predictor

import (
	"bytes"
	"container/heap"
	"context"
	"fmt"
	"io/ioutil"
	"math"
	"sort"
	"strings"

	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/tensorflow"
	"github.com/rai-project/tracer"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	gotensor "gorgonia.org/tensor"
)

// ImageCaptioningPredictor struct
type ImageCaptioningPredictor struct {
	*ImagePredictor
	CNNInputLayer         string
	seqSentenceInputLayer string
	seqStateInputLayer    string
	CNNStateOutputLayer   string
	seqSoftmaxLayer       string
	seqStateOutputLayer   string
	vocabulary            vocabularyT
	captions              interface{}
}

// NewImageCaptioningPredictor is a constructor for ImageCaptioningPredictor.
func NewImageCaptioningPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(InstanceSegmentationPredictor)

	return predictor.Load(ctx, model, opts...)
}

// Load ...
func (self *ImageCaptioningPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}

	p := &ImageCaptioningPredictor{
		ImagePredictor: pred,
	}

	model, err := ioutil.ReadFile(p.GetGraphPath())
	if err != nil {
		return nil, errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
	}
	modelReader := bytes.NewReader(model)

	p.CNNInputLayer, err = p.GetInputLayerName(modelReader, "image_input_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get image input layer name")
	}

	p.seqSentenceInputLayer, err = p.GetInputLayerName(modelReader, "sentence_input_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get intermediate sentence input layer name")
	}

	p.seqStateInputLayer, err = p.GetInputLayerName(modelReader, "state_input_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get intermediate state input layer name")
	}

	p.CNNStateOutputLayer, err = p.GetOutputLayerName(modelReader, "initial_state_output_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the initial state output layer name")
	}
	p.seqSoftmaxLayer, err = p.GetOutputLayerName(modelReader, "softmax_output_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the intermediate softmax output layer name")
	}
	p.seqStateOutputLayer, err = p.GetOutputLayerName(modelReader, "state_output_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the intermediate state output layer name")
	}

	return p, nil
}

// Predict ...
func (p *ImageCaptioningPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	if data == nil {
		return errors.New("input data nil")
	}
	input, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	session := p.tfSession
	graph := p.tfGraph

	tensor, err := makeTensorFromGoTensors(input)
	if err != nil {
		return err
	}

	sessionSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict",
		opentracing.Tags{
			"evaluation_trace_level": p.TraceLevel(),
		})

	initialState, err := session.Run(ctx,
		map[tf.Output]*tf.Tensor{
			graph.Operation(p.CNNInputLayer).Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation(p.CNNStateOutputLayer).Output(0),
		},
		nil,
		p.runOptions(),
		p.GetGraphPath(),
	)

	beamSize := 3
	maxCaptionLength := 20
	lengthNormalizationFactor := 0.0
	vocabularyFile := "./word_counts.txt"
	p.vocabulary = constructVocabulary(vocabularyFile)

	vocab := p.vocabulary
	// no actual meaning. just to squeeze the 0th dimension.
	initialStateArray, _ := tf.NewTensor(initialState[0].Value().([][]float32)[0])

	initialBeam := caption{
		sentence: []int64{vocab.startID},
		state:    initialStateArray,
		logprob:  0.0,
		score:    0.0,
	}

	partialCaptions := &topN{n: beamSize}
	heap.Init(partialCaptions)
	partialCaptions.PushTopN(initialBeam)

	completeCaptions := &topN{n: beamSize}
	heap.Init(completeCaptions)

	intermediateInputFeed := graph.Operation("input_feed")
	intermediateStateFeed := graph.Operation("lstm/state_feed")
	softmaxOp := graph.Operation("softmax")
	stateOp := graph.Operation("lstm/state")

	for i := 0; i < maxCaptionLength-1; i++ {
		partialCaptionsList := partialCaptions.Extract(false)
		// partialCaptions.Reset()

		inputFeed := []int64{}
		stateFeed := []*tf.Tensor{}
		for _, partialCaption := range partialCaptionsList {
			inputFeed = append(inputFeed, partialCaption.sentence[len(partialCaption.sentence)-1])
			stateFeed = append(stateFeed, partialCaption.state)
		}
		inputTensor, err := tf.NewTensor(inputFeed)
		if err != nil {
			fmt.Println("inputTensor error:", err)
		}
		stateTensor := batchify(stateFeed)

		intermediateOutput, err := session.Run(ctx,
			map[tf.Output]*tf.Tensor{
				intermediateInputFeed.Output(0): inputTensor,
				intermediateStateFeed.Output(0): stateTensor,
			},
			[]tf.Output{
				softmaxOp.Output(0),
				stateOp.Output(0),
			},
			nil,
			p.runOptions(),
			p.GetGraphPath(),
		)
		if err != nil {
			fmt.Println("intermediate session run error:", err)
			log.Fatal(err)
		}

		softmaxOutput := intermediateOutput[0].Value().([][]float32)
		stateOutput := intermediateOutput[1].Value().([][]float32)

		// fmt.Printf("%d: inputFeed: %v\n", i, inputFeed)
		for j, partialCaption := range partialCaptionsList {
			wordProbabilities := softmaxOutput[j]
			state := stateOutput[j]

			wordLen := len(wordProbabilities)
			idxs := []int64{}
			for idx := int64(0); idx < int64(wordLen); idx++ {
				idxs = append(idxs, idx)
			}
			arg := ArgSort{Args: wordProbabilities, Idxs: idxs}
			sort.Sort(arg)
			mostLikelyWords := arg.Idxs[wordLen-beamSize : wordLen]
			mostLikelyWordsProb := arg.Args[wordLen-beamSize : wordLen]

			// fmt.Println(mostLikelyWords, mostLikelyWordsProb)

			for k := len(mostLikelyWords) - 1; k >= 0; k-- {
				w := mostLikelyWords[k]
				p := mostLikelyWordsProb[k]
				if p < 1e-12 {
					continue
				}

				sentence := make([]int64, len(partialCaption.sentence))
				copy(sentence, partialCaption.sentence)
				sentence = append(sentence, w)
				logprob := partialCaption.logprob + float32(math.Log(float64(p)))
				score := logprob
				// fmt.Println("sentence:", sentence)
				// fmt.Println("word:", w, "->", p)
				// fmt.Printf("sentence: %v, prob: %f\n", sentence, logprob)
				newStateTensor, _ := tf.NewTensor(state)
				// fmt.Println("inner stateTensor shape:", stateTensor.Shape())

				if w == vocab.endID {
					if lengthNormalizationFactor > 0 {
						score /= float32(math.Pow(float64(len(sentence)), lengthNormalizationFactor))
					}
					beam := caption{
						sentence: sentence,
						state:    newStateTensor,
						logprob:  logprob,
						score:    score,
					}
					completeCaptions.PushTopN(beam)
				} else {
					beam := caption{
						sentence: sentence,
						state:    newStateTensor,
						logprob:  logprob,
						score:    score,
					}
					partialCaptions.PushTopN(beam)
					// fmt.Printf("heap: ")
					// for _, c := range partialCaptions.captionHeap {
					// 	fmt.Printf("%v", c.sentence)
					// }
					// fmt.Printf("\n")
				}

				if partialCaptions.Len() == 0 {
					break
				}
			}
		}

		// fmt.Printf("partialCaptions after round %d: ", i)
		// for _, c := range partialCaptions.captionHeap {
		// 	fmt.Printf("%v", c.sentence)
		// }
		// fmt.Printf("\n\n")
		// if i == 2 {
		// 	break
		// }
	}

	if completeCaptions.Len() == 0 {
		completeCaptions = partialCaptions
	}

	captions := completeCaptions.Extract(true)

	p.cuptiClose()

	sessionSpan.Finish()

	if err != nil {
		return errors.Wrapf(err, "failed to perform session.Run")
	}

	p.captions = captions

	return nil
}

// ReadPredictedFeatures ...
func (p *ImageCaptioningPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	captions, ok := p.captions.([]caption)
	if !ok {
		return nil, errors.New("output is not of type []caption")
	}
	vocab := p.vocabulary
	words := []string{}
	sentences := []string{}
	// IDs := []int64{}
	probabilities := []float64{}
	for i := len(captions) - 1; i >= 0; i-- {
		//print the final result
		caption := captions[i]
		for _, wordID := range caption.sentence {
			words = append(words, vocab.reverseVocab[wordID])
			// IDs = append(IDs, wordID)
		}
		predSentence := strings.Join(words[1:len(words)-1], " ")
		sentences = append(sentences, predSentence)
		probabilities = append(probabilities, math.Exp(float64(caption.logprob)))
		// fmt.Printf("%d) %s (p=%f)   ", i, predSentence, math.Exp(float64(caption.logprob)))
		// fmt.Println(IDs)
		words = nil
		// IDs = nil
	}

	return p.CreateCaptioningFeatures(ctx, sentences, probabilities)
}

// Modality ...
func (p ImageCaptioningPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageCaptioningModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := tensorflow.FrameworkManifest
		agent.AddPredictor(framework, &ImageCaptioningPredictor{
			ImagePredictor: &ImagePredictor{
				ImagePredictor: common.ImagePredictor{
					Base: common.Base{
						Framework: framework,
					},
				},
			},
		})
	})
}
