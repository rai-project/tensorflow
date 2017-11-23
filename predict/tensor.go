package predict

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
// #include <string.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"unsafe"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"golang.org/x/net/context"
)

var tftypes = []struct {
	typ      reflect.Type
	dataType C.TF_DataType
}{
	{reflect.TypeOf(float32(0)), C.TF_FLOAT},
	{reflect.TypeOf(float64(0)), C.TF_DOUBLE},
	{reflect.TypeOf(int32(0)), C.TF_INT32},
	{reflect.TypeOf(uint8(0)), C.TF_UINT8},
	{reflect.TypeOf(int16(0)), C.TF_INT16},
	{reflect.TypeOf(int8(0)), C.TF_INT8},
	{reflect.TypeOf(""), C.TF_STRING},
	{reflect.TypeOf(complex(float32(0), float32(0))), C.TF_COMPLEX64},
	{reflect.TypeOf(int64(0)), C.TF_INT64},
	{reflect.TypeOf(false), C.TF_BOOL},
	{reflect.TypeOf(uint16(0)), C.TF_UINT16},
	{reflect.TypeOf(complex(float64(0), float64(0))), C.TF_COMPLEX128},
	// TODO(apassos): support DT_RESOURCE representation in go.
}

func NewTensor(ctx context.Context, data [][]float32, shape []int64) (*tf.Tensor, error) {
	val := reflect.ValueOf(data)
	dataType, err := dataTypeOf(val)
	if err != nil {
		return nil, err
	}

	nflattened := numElements(shape)
	nbytes := typeOf(dataType, nil).Size() * uintptr(nflattened)

	if dataType == tf.String {
		return nil, errors.New("string not support type for new tensor")
	}
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	t := &Tensor{
		c:     C.TF_AllocateTensor(C.TF_DataType(dataType), shapePtr, C.int(len(shape)), C.size_t(nbytes)),
		shape: shape,
	}

	runtime.SetFinalizer(t, (*Tensor).finalize)
	raw := tensorData(t.c)
	buf := bytes.NewBuffer(raw[:0:len(raw)])

	if err := encodeTensor(buf, val); err != nil {
		return nil, err
	}
	if uintptr(buf.Len()) != nbytes {
		return nil, bug("NewTensor incorrectly calculated the size of a tensor with type %v and shape %v as %v bytes instead of %v", dataType, shape, nbytes, buf.Len())
	}

	return (*tf.Tensor)(unsafe.Pointer(t)), nil
}

// encodeTensor writes v to the specified buffer using the format specified in
// c_api.h. Use stringEncoder for String tensors.
func encodeTensor(w *bytes.Buffer, v reflect.Value) error {
	if v.Kind() == reflect.Slice && v.Len() > 0 &&
		v.Type().Elem().Kind() == reflect.Slice &&
		v.Index(0).Len() > 0 &&
		v.Index(0).Type().Elem().Kind() == reflect.Float32 {
		data, ok := v.Interface().([][]float32)
		if !ok {
			return errors.New("expecting a [][]float32 type")
		}
		log.Info("encoding tensor using [][]float32")
		for _, row := range data {
			for _, elem := range row {
				var b [4]byte
				val := math.Float32bits(elem)
				nativeEndian.PutUint32(b[:], val)
				w.Write(b[:])
			}
		}
		return nil
	}
	switch v.Kind() {
	case reflect.Bool:
		b := byte(0)
		if v.Bool() {
			b = 1
		}
		if err := w.WriteByte(b); err != nil {
			return err
		}
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		if err := binary.Write(w, nativeEndian, v.Interface()); err != nil {
			return err
		}

	case reflect.Array, reflect.Slice:
		// If slice elements are slices, verify that all of them have the same size.
		// Go's type system makes that guarantee for arrays.
		if v.Len() > 0 && v.Type().Elem().Kind() == reflect.Slice {
			expected := v.Index(0).Len()
			for i := 1; i < v.Len(); i++ {
				if v.Index(i).Len() != expected {
					return fmt.Errorf("mismatched slice lengths: %d and %d", v.Index(i).Len(), expected)
				}
			}
		}

		for i := 0; i < v.Len(); i++ {
			err := encodeTensor(w, v.Index(i))
			if err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v -- %v", v.Type(), v.Kind())
	}
	return nil
}

func tensorData(c *C.TF_Tensor) []byte {
	// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	cbytes := C.TF_TensorData(c)
	length := int(C.TF_TensorByteSize(c))
	slice := (*[1 << 30]byte)(unsafe.Pointer(cbytes))[:length:length]
	return slice
}

// typeOf converts from a DataType and Shape to the equivalent Go type.
func typeOf(dt tf.DataType, shape []int64) reflect.Type {
	var ret reflect.Type
	for _, t := range tftypes {
		if dt == tf.DataType(t.dataType) {
			ret = t.typ
			break
		}
	}
	if ret == nil {
		panic(bug("DataType %v is not supported", dt))
	}
	for range shape {
		ret = reflect.SliceOf(ret)
	}
	return ret
}

func bug(format string, args ...interface{}) error {
	return fmt.Errorf("BUG: Please report at https://github.com/tensorflow/tensorflow/issues with the note: Go TensorFlow %v: %v", tf.Version(), fmt.Sprintf(format, args...))
}

func numElements(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// dataTypeOf returns the data type of the Tensor
// corresponding to a Go type.
func dataTypeOf(val reflect.Value) (dt tf.DataType, err error) {
	typ := val.Type()

	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		typ = typ.Elem()
	}
	for _, t := range tftypes {
		if typ.Kind() == t.typ.Kind() {
			return tf.DataType(t.dataType), nil
		}
	}
	return dt, fmt.Errorf("unsupported type %v when getting data type of tensor", typ)
}

// nativeEndian is the byte order for the local platform. Used to send back and
// forth Tensors with the C API. We test for endianness at runtime because
// some architectures can be booted into different endian modes.
var nativeEndian binary.ByteOrder

func init() {
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = uint16(0xABCD)

	switch buf {
	case [2]byte{0xCD, 0xAB}:
		nativeEndian = binary.LittleEndian
	case [2]byte{0xAB, 0xCD}:
		nativeEndian = binary.BigEndian
	default:
		panic("Could not determine native endianness.")
	}
}
