package graph

import (
	"encoding/json"
	"io/ioutil"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
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

type opInfo struct {
	Name   string `json:"name,omitempty"`
	Schema struct {
		Attributes []struct {
			AllowedValues interface{} `json:"allowedValues,omitempty"`
			Name          string      `json:"name,omitempty"`
			Type          interface{} `json:"type,omitempty"`
			Default       interface{} `json:"default,omitempty"`
			Description   string      `json:"description,omitempty"`
		} `json:"attributes,omitempty"`
		Description string `json:"description,omitempty"`
		Inputs      []struct {
			Description string `json:"description,omitempty"`
			IsRef       bool   `json:"isRef,omitempty"`
			Name        string `json:"name,omitempty"`
			TypeAttr    string `json:"typeAttr,omitempty"`
		} `json:"inputs,omitempty"`
		Outputs []struct {
			Description string `json:"description,omitempty"`
			IsRef       bool   `json:"isRef,omitempty"`
			Name        string `json:"name,omitempty"`
			TypeAttr    string `json:"typeAttr,omitempty"`
		} `json:"outputs,omitempty"`
		Summary string `json:"summary,omitempty"`
	} `json:"schema,omitempty"`
}

var OpInfo = []opInfo{}

type Graph struct {
	*tf.GraphDef
}

func New(path string) (*Graph, error) {
	if !com.IsFile(path) {
		return nil, errors.Errorf("the file %v was not found", path)
	}
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
				Value: &tf.AttrValue_Placeholder{
					Placeholder: cat,
				},
			}
		}
		if OpInfo != nil {
			for _, op := range OpInfo {
				if op.Name == nd.GetOp() {
					bts, err := json.Marshal(op)
					if err != nil {
						continue
					}
					nd.Attr["op_info"] = &tf.AttrValue{
						Value: &tf.AttrValue_Placeholder{
							Placeholder: string(bts),
						},
					}
					break
				}
			}
		}
	}
	return json.Marshal(g.GraphDef)
}

func init() {
	metadata := MustAsset("_fixtures/tf-metadata.json")
	err := json.Unmarshal(metadata, &OpInfo)
	if err != nil {
		pp.Println(err.Error())
		OpInfo = nil
	}
}
