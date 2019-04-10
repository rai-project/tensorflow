package tensorboard

import (
	"fmt"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestTensorBoard(t *testing.T) {
	step := 0
	w := &tfsum.Writer{Dir: "./tf-log", Name: "train"}
	var s *tf.Session
	var g *tf.Graph
	var input *tf.Tensor
	sum, err := s.Run(
		map[tf.Output]*tf.Tensor{
			g.Operation("input").Output(0): input,
		},
		[]tf.Output{
			g.Operation("MergeSummary/MergeSummary").Output(0),
		},
		[]*tf.Operation{
			g.Operation("train_step"),
		})
	if err != nil {
		fmt.Println(err)
	}
	err = w.AddEvent(sum[0].Value().(string), int64(step))
	if err != nil {
		fmt.Println(err)
	}
}
