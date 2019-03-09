package predictor

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
  "unsafe"
	"reflect"

	forceexport "github.com/alangpierce/go-forceexport"
	"github.com/k0kubun/pp"
	// "github.com/spance/go-callprivate/private"
)

var (
	newTensorFromC func(c *C.TF_Tensor) *Tensor
)

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

func init() {
	const tfPackagePath = "github.com/tensorflow/tensorflow/tensorflow/go"
	forceexport.GetFunc(&newTensorFromC, tfPackagePath+".newTensorFromC")
  if false {
    pp.Println("init")
  }
}
