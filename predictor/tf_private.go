package predictor

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"github.com/k0kubun/pp"
	"reflect"
	"unsafe"
	"runtime"
	// "github.com/spance/go-callprivate/private"
)

func tensor_newTensorFromC(c *C.TF_Tensor) *Tensor {
	var shape []int64
	if ndims := int(C.TF_NumDims(c)); ndims > 0 {
		shape = make([]int64, ndims)
	}
	for i := range shape {
		shape[i] = int64(C.TF_Dim(c, C.int(i)))
	}

	t := new(Tensor)

	field_c := reflect.ValueOf(t).Elem().FieldByName("c")
	reflect.NewAt(field_c.Type(), unsafe.Pointer(field_c.UnsafeAddr())).Elem().Set(reflect.ValueOf(c).Convert(field_c.Type()))

	field_shape := reflect.ValueOf(t).Elem().FieldByName("shape")
	reflect.NewAt(field_shape.Type(), unsafe.Pointer(field_shape.UnsafeAddr())).Elem().Set(reflect.ValueOf(shape).Convert(field_shape.Type()))

	runtime.SetFinalizer(t, tensor_finalize)
	return t
}

func tensor_finalize(t *Tensor) { C.TF_DeleteTensor(tensorPtrC(t)) }

func outputC(p Output) C.TF_Output {
	if p.Op == nil {
		// Attempt to provide a more useful panic message than "nil
		// pointer dereference".
		panic("nil-Operation. If the Output was created with a Scope object, see Scope.Err() for details.")
	}

	return C.TF_Output{oper: operationPtrC(p.Op), index: C.int(p.Index)}
}

func tensorPtrC(t *Tensor) *C.TF_Tensor {
	fld := reflect.Indirect(reflect.ValueOf(t)).FieldByName("c")
	if fld.CanInterface() {
		return fld.Interface().(*C.TF_Tensor)
	}

	ptr := unsafe.Pointer(fld.UnsafeAddr())
	e := (**C.TF_Tensor)(ptr)
	return *e
}

func operationPtrC(o *Operation) *C.TF_Operation {
	fld := reflect.Indirect(reflect.ValueOf(o)).FieldByName("c")
	if fld.CanInterface() {
		return fld.Interface().(*C.TF_Operation)
	}

	ptr := unsafe.Pointer(fld.UnsafeAddr())
	e := (**C.TF_Operation)(ptr)
	return *e
}

func graphPtrC(g *Graph) *C.TF_Graph {
	fld := reflect.Indirect(reflect.ValueOf(g)).FieldByName("c")
	if fld.CanInterface() {
		return fld.Interface().(*C.TF_Graph)
	}

	ptr := unsafe.Pointer(fld.UnsafeAddr())
	e := (**C.TF_Graph)(ptr)
	return *e
}

func tensorData(c *C.TF_Tensor) []byte {
	// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	cbytes := C.TF_TensorData(c)
	if cbytes == nil {
		return nil
	}
	length := int(C.TF_TensorByteSize(c))
	slice := (*[1 << 30]byte)(unsafe.Pointer(cbytes))[:length:length]
	return slice
}

func init() {
	if false {
		pp.Println("init")
	}
}
