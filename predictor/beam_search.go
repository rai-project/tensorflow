package predictor

import (
	"bufio"
	"container/heap"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type caption struct {
	sentence []int64
	state    *tf.Tensor
	logprob  float32
	score    float32
}

type topN struct {
	captionHeap []caption
	n           int
}

func (t topN) Len() int {
	return len(t.captionHeap)
}

func (t topN) Less(i, j int) bool { return t.captionHeap[i].score < t.captionHeap[j].score }
func (t topN) Swap(i, j int) {
	t.captionHeap[i], t.captionHeap[j] = t.captionHeap[j], t.captionHeap[i]
}

func (t *topN) Push(x interface{}) {
	t.captionHeap = append(t.captionHeap, x.(caption))
}

func (t *topN) Pop() interface{} {
	old := t.captionHeap
	currLength := len(old)
	x := old[currLength-1]
	t.captionHeap = old[0 : currLength-1]
	return x
}

func (t *topN) PushTopN(x interface{}) {
	if t.Len() < t.n {
		heap.Push(t, x.(caption))
	} else {
		heap.Push(t, x.(caption))
		heap.Pop(t)
	}
}

func (t *topN) Extract(sort bool) []caption {
	result := []caption{}
	if sort {
		for t.Len() > 0 {
			result = append(result, heap.Pop(t).(caption))
		}
	} else {
		result = make([]caption, t.Len())
		copy(result, t.captionHeap)
		t.captionHeap = nil
	}

	return result
}

func (t *topN) Reset() {
	t.captionHeap = []caption{}
}

type vocabularyT struct {
	vocab        map[string]int64
	reverseVocab map[int64]string
	startID      int64
	endID        int64
	unknownID    int64
}

func constructVocabulary(vocabFile string) (vocabulary vocabularyT) {
	vocab := make(map[string]int64)
	reverseVocab := make(map[int64]string)
	startWord := "<S>"
	endWord := "</S>"
	unknownWord := "<UNK>"

	file, err := os.Open(vocabFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	id := 0
	for scanner.Scan() {
		// fmt.Println(scanner.Text())
		wordPair := strings.Split(scanner.Text(), " ")
		word := wordPair[0]
		// id, _ := strconv.Atoi(wordPair[1])
		vocab[word] = int64(id)
		reverseVocab[int64(id)] = word
		id++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	startID := vocab[startWord]
	endID := vocab[endWord]
	unknownID := vocab[unknownWord]

	vocabulary = vocabularyT{
		vocab:        vocab,
		reverseVocab: reverseVocab,
		startID:      startID,
		endID:        endID,
		unknownID:    unknownID}

	return vocabulary
}

// ArgSort ...
type ArgSort struct {
	Args []float32
	Idxs []int64
}

// Implement sort.Interface Len
func (s ArgSort) Len() int { return len(s.Args) }

// Implement sort.Interface Less
func (s ArgSort) Less(i, j int) bool { return s.Args[i] < s.Args[j] }

// Implment sort.Interface Swap
func (s ArgSort) Swap(i, j int) {
	// swap value
	s.Args[i], s.Args[j] = s.Args[j], s.Args[i]
	// swap index
	s.Idxs[i], s.Idxs[j] = s.Idxs[j], s.Idxs[i]
}

func batchify(tensors []*tf.Tensor) (packedTensor *tf.Tensor) {
	s := op.NewScope()
	var pack []tf.Output
	for i := 0; i < len(tensors); i++ {
		input := op.Placeholder(s.SubScope("input"), tf.Float)
		pack = append(pack, input)
	}
	output := op.Pack(s, pack)

	graph, err := s.Finalize()
	if err != nil {
		panic(err.Error())
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err.Error())
	}
	defer session.Close()

	inputFeed := make(map[tf.Output]*tf.Tensor)
	for i := 0; i < len(tensors); i++ {
		inputFeed[pack[i]] = tensors[i]
	}
	batchOutput, err := session.Run(
		inputFeed,
		[]tf.Output{output},
		nil)
	if err != nil {
		fmt.Println("input image conversion to tensor error", err)
	}

	return batchOutput[0]
}

func beamSearch(beamSize, maxCaptionLength int, lengthNormalizationFactor float64, vocabulary string, p *ImageCaptioningPredictor, initialState []*tf.Tensor, ops ...string) (captions []caption) {
	vocab := constructVocabulary(vocabulary)

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

	// intermediateInputFeed := graph.Operation("input_feed")
	// intermediateStateFeed := graph.Operation("lstm/state_feed")
	// softmaxOp := graph.Operation("softmax")
	// stateOp := graph.Operation("lstm/state")

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

		// fmt.Printf("inputTensor shape: %v\n", inputTensor.Shape())

		session := p.tfSession
		graph := p.tfGraph

		intermediateOutput, err := session.Run(
			map[tf.Output]*tf.Tensor{
				graph.Operation(ops[0]).Output(0): inputTensor,
				graph.Operation(ops[1]).Output(0): stateTensor,
			},
			[]tf.Output{
				graph.Operation(ops[2]).Output(0),
				graph.Operation(ops[3]).Output(0),
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

	captions = completeCaptions.Extract(true)

	return captions
}
