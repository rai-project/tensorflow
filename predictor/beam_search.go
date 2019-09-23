package predictor

import (
	"bufio"
	"container/heap"
	"fmt"
	"os"
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
