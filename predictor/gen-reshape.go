// This file was automatically generated by genny.
// Any changes will be lost if this file is regenerated.
// see https://github.com/cheekybits/genny

//go:generate go get github.com/cheekybits/genny

package predictor

import tf "github.com/tensorflow/tensorflow/tensorflow/go"

func flattenedUint8ToTensor(data []uint8, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint8, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]uint8, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint8, W)
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

func Uint8SliceToTensor(data [][]uint8, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint8, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]uint8, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint8, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedUint16ToTensor(data []uint16, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint16, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]uint16, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint16, W)
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

func Uint16SliceToTensor(data [][]uint16, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint16, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]uint16, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint16, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedUint32ToTensor(data []uint32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint32, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]uint32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint32, W)
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

func Uint32SliceToTensor(data [][]uint32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint32, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]uint32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint32, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedUint64ToTensor(data []uint64, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint64, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]uint64, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint64, W)
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

func Uint64SliceToTensor(data [][]uint64, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]uint64, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]uint64, H)
		for h := int64(0); h < H; h++ {
			th := make([][]uint64, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedInt8ToTensor(data []int8, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int8, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]int8, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int8, W)
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

func Int8SliceToTensor(data [][]int8, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int8, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]int8, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int8, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedInt16ToTensor(data []int16, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int16, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]int16, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int16, W)
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

func Int16SliceToTensor(data [][]int16, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int16, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]int16, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int16, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedInt32ToTensor(data []int32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int32, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]int32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int32, W)
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

func Int32SliceToTensor(data [][]int32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int32, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]int32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int32, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedInt64ToTensor(data []int64, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int64, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]int64, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int64, W)
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

func Int64SliceToTensor(data [][]int64, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]int64, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]int64, H)
		for h := int64(0); h < H; h++ {
			th := make([][]int64, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedFloat32ToTensor(data []float32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]float32, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]float32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]float32, W)
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

func Float32SliceToTensor(data [][]float32, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]float32, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]float32, H)
		for h := int64(0); h < H; h++ {
			th := make([][]float32, W)
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

//go:generate go get github.com/cheekybits/genny

func flattenedFloat64ToTensor(data []float64, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]float64, N)
	for n := int64(0); n < N; n++ {
		tn := make([][][]float64, H)
		for h := int64(0); h < H; h++ {
			th := make([][]float64, W)
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

func Float64SliceToTensor(data [][]float64, shape []int64) (*tf.Tensor, error) {
	N, H, W, C := shape[0], shape[1], shape[2], shape[3]
	tensor := make([][][][]float64, N)
	for n := int64(0); n < N; n++ {
		ndata := data[n]
		tn := make([][][]float64, H)
		for h := int64(0); h < H; h++ {
			th := make([][]float64, W)
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
