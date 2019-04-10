//+build ignore

package main

import (
	"fmt"
	"os"

	"github.com/rai-project/tensorflow/graph"
)

func main() {
	g, err := graph.New(os.Args[1])
	if err != nil {
		panic(err)
	}
	bts, err := g.MarshalJSON()
	if err != nil {
		panic(err)
	}
	fmt.Println(string(bts))
}
