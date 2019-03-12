//go:generate go get github.com/cheekybits/genny
//go:generate genny -in=$GOFILE -out=gen-$GOFILE gen "ElementType=uint8,uint16,uint32,uint64,int8,int16,int32,int64,float32,float64"

package predictor

import (
	"github.com/cheekybits/genny/generic"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type ElementType generic.Type

func flattenedElementTypeToTensor(data []ElementType, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]ElementType, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]ElementType, H)
		for h := int64(0); h < H; h++ {
			th := make([][]ElementType, W)
			for w := int64(0); w < W; w++ {
				offset := C * (n*W*H + W*h + w)
				tw := data[offset : offset+C]
				th[w] = tw
			}
			tn[h] = th
		}
		tensor[n] = tn
	}
	return tf.NewTensor(tensor)
}

func ElementTypeSliceToTensor(data [][]ElementType, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]ElementType, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]ElementType, H)
		for h := int64(0); h < H; h++ {
			th := make([][]ElementType, W)
			for w := int64(0); w < W; w++ {
				offset := C * (W*h + w)
				tw := ndata[offset : offset+C]
				th[w] = tw
			}
			tn[h] = th
		}
		tensor[n] = tn
	}
	return tf.NewTensor(tensor)
}
