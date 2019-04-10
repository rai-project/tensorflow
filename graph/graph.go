package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	tf "github.com/rai-project/tensorflow"
	// tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var Categories = map[string]string{
	"Const":                            "Constant",
	"Conv2D":                           "Layer",
	"BiasAdd":                          "Layer",
	"DepthwiseConv2dNative":            "Layer",
	"Relu":                             "Activation",
	"Relu6":                            "Activation",
	"Softmax":                          "Activation",
	"Sigmoid":                          "Activation",
	"LRN":                              "Normalization",
	"MaxPool":                          "Pool",
	"MaxPoolV2":                        "Pool",
	"AvgPool":                          "Pool",
	"Reshape":                          "Shape",
	"Squeeze":                          "Shape",
	"ConcatV2":                         "Tensor",
	"Split":                            "Tensor",
	"Dequantize":                       "Tensor",
	"Identity":                         "Control",
	"Variable":                         "Control",
	"VariableV2":                       "Control",
	"Assign":                           "Control",
	"BatchNormWithGlobalNormalization": "Normalization",
	"FusedBatchNorm":                   "Normalization",
}

type Graph struct {
	*tf.GraphDef
}

func New(path string) (*Graph, error) {

	model, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	graph := &tf.GraphDef{}
	err = graph.Unmarshal(model)

	return &Graph{
		GraphDef: graph}, err
}

func (g *Graph) MarshalJSON() ([]byte, error) {
	for _, nd := range g.Node {
		nd.Attr = map[string]*tf.AttrValue{}
		if cat, ok := Categories[nd.GetName()]; ok {
			nd.Attr["category"] = &tf.AttrValue{
				Value: &tf.AttrValue_S{
					S: []byte(cat),
				},
			}
		}
	}
	return json.Marshal(g.GraphDef)
}

func main() {
	g, err := New(os.Args[1])
	if err != nil {
		panic(err)
	}
	bts, err := g.MarshalJSON()
	if err != nil {
		panic(err)
	}
	fmt.Println(string(bts))
}
