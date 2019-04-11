package graph

import (
	"encoding/json"
	"io/ioutil"
	"sync"

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

type TensorInfo struct {
	Name              string  `json:"name,omitempty"`
	DataType          string  `json:"data_type,omitempty"`
	DataTypeByteCount int64   `json:"data_type_byte_count,omitempty"`
	ByteCount         int64   `json:"byte_count,omitempty"`
	Dims              []int64 `json:"dims,omitempty"`
	OpName            string  `json:"op_name,omitempty"`
}

type Graph struct {
	*tf.GraphDef
	TensorInfos []TensorInfo `json:"tensor_infos,omitempty"`
	NumBytes    int64        `json:"num_parameters,omitempty"`
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
		GraphDef:    graph,
		TensorInfos: []TensorInfo{},
	}, err
}

func tensorShape(tensor *tf.TensorShapeProto) []int64 {
	res := make([]int64, len(tensor.Dim))
	for ii, dim := range tensor.Dim {
		res[ii] = dim.Size_
	}
	return res
}

func prod(lst []int64) int64 {
	accum := int64(1)
	for _, elem := range lst {
		accum *= elem
	}
	return accum
}

func (g *Graph) MarshalJSON() ([]byte, error) {
	initOpInfo()
	for _, nd := range g.Node {
		currentNumParameters := int64(0)
		if nd.Attr == nil {
			nd.Attr = map[string]*tf.AttrValue{}
		}
		for name, attr := range nd.Attr {
			if name != "value" {
				continue
			}
			tensor := attr.Value.(*tf.AttrValue_Tensor).Tensor
			g.NumBytes += int64(len(tensor.TensorContent))

			dtype := DataType(tensor.Dtype)
			byteCount := int64(dtype.ByteCount())

			dims := tensorShape(tensor.TensorShape)
			g.TensorInfos = append(g.TensorInfos, TensorInfo{
				Name:              nd.GetName(),
				DataType:          dtype.String(),
				DataTypeByteCount: byteCount,
				Dims:              dims,
				ByteCount:         prod(dims) * byteCount,
				OpName:            nd.GetOp(),
			})

			attr.Value.(*tf.AttrValue_Tensor).Tensor.TensorContent = nil

			break
		}

		for _, attr := range nd.Attr {
			if s, ok := attr.Value.(*tf.AttrValue_Tensor); ok {
				s.Tensor.TensorContent = nil
				s.Tensor.FloatVal = nil
			}
			if s, ok := attr.Value.(*tf.AttrValue_List); ok {
				s.List = nil
			}
		}

		g.NumBytes += currentNumParameters

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

		// for debugging
		// if _, err := json.Marshal(nd); err != nil {
		// 	pp.Println(nd)
		// 	return nil, err
		// }
	}

	return json.Marshal(&struct {
		*tf.GraphDef
		TensorInfos []TensorInfo `json:"tensor_infos,omitempty"`
		NumBytes    int64        `json:"num_parameters,omitempty"`
	}{
		g.GraphDef,
		g.TensorInfos,
		g.NumBytes,
	})
}

func initOpInfo() {
	var once sync.Once
	once.Do(func() {
		metadata := MustAsset("_fixtures/tf-metadata.json")
		err := json.Unmarshal(metadata, &OpInfo)
		if err != nil {
			pp.Println(err.Error())
			OpInfo = nil
		}
	})
}
