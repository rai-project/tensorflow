/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package predictor

// #include <stdlib.h>
// #include <string.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
  "unsafe"
	protobuf "github.com/golang/protobuf/proto"
	proto "github.com/rai-project/tensorflow"
)

// Session drives a TensorFlow graph computation.
//
// When a Session is created with a given target, a new Session object is bound
// to the universe of resources specified by that target. Those resources are
// available to this session to perform computation described in the GraphDef.
// After creating the session with a graph, the caller uses the Run() API to
// perform the computation and potentially fetch outputs as Tensors.
// A Session allows concurrent calls to Run().
type Session struct {
	c *C.TF_Session

	// For ensuring that:
	// - Close() blocks on all Run() calls to complete.
	// - Close() can be called multiple times.
	wg sync.WaitGroup
	mu sync.Mutex
}

// NewSession creates a new execution session with the associated graph.
// options may be nil to use the default options.
func NewSession(graph *Graph, options *SessionOptions) (*Session, error) {
	status := newStatus()
	cOpt, doneOpt, err := options.c()
	defer doneOpt()
	if err != nil {
		return nil, err
  }
	cSess := C.TF_NewSession(graphPtrC(graph), cOpt, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}

	s := &Session{c: cSess}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return s, nil
}

// Device structure contains information about a device associated with a session, as returned by ListDevices()
type Device struct {
	Name, Type       string
	MemoryLimitBytes int64
}

// String describes d and implements fmt.Stringer.
func (d Device) String() string {
	memStr := "no memory limit"
	if d.MemoryLimitBytes >= 0 {
		memStr = fmt.Sprintf("memory limit %d bytes", d.MemoryLimitBytes)
	}
	return fmt.Sprintf("(Device: name \"%s\", type %s, %s)", d.Name, d.Type, memStr)
}

func deviceSliceFromDeviceList(list *C.TF_DeviceList) ([]Device, error) {
	var devices []Device
	status := newStatus()

	for i := 0; i < int(C.TF_DeviceListCount(list)); i++ {
		name := C.TF_DeviceListName(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListName(index=%d) failed: %v", i, err)
		}

		deviceType := C.TF_DeviceListType(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListType(index=%d) failed: %v", i, err)
		}

		memoryLimitBytes := C.TF_DeviceListMemoryBytes(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListMemoryBytes(index=%d) failed: %v", i, err)
		}

		device := Device{
			Name:             C.GoString(name),
			Type:             C.GoString(deviceType),
			MemoryLimitBytes: int64(memoryLimitBytes),
		}

		devices = append(devices, device)
	}

	return devices, nil
}

// ListDevices returns the list of devices associated with a Session.
func (s *Session) ListDevices() ([]Device, error) {
	status := newStatus()
	devicesList := C.TF_SessionListDevices(s.c, status.c)
	if err := status.Err(); err != nil {
		return nil, fmt.Errorf("SessionListDevices() failed: %v", err)
	}
	defer C.TF_DeleteDeviceList(devicesList)
	return deviceSliceFromDeviceList(devicesList)
}

// Run the graph with the associated session starting with the supplied feeds
// to compute the value of the requested fetches. Runs, but does not return
// Tensors for operations specified in targets.
//
// On success, returns the fetched Tensors in the same order as supplied in
// the fetches argument. If fetches is set to nil, the returned Tensor fetches
// is empty.


func (s *Session) Run(ctx context.Context, feeds map[Output]*Tensor, fetches []Output, targets []*Operation, runOpts *proto.RunOptions) ([]*Tensor, error) {
	s.mu.Lock()
	if s.c == nil {
		s.mu.Unlock()
		return nil, errors.New("session is closed")
	}
	s.wg.Add(1)
	s.mu.Unlock()
	defer s.wg.Done()

	c := newCRunArgs(feeds, fetches, targets)
  status := newStatus()
  
	var runOptsBuf *C.TF_Buffer
	var runMetaData *C.TF_Buffer

	if runOpts != nil {
		runOptsBuf = C.TF_NewBuffer()
		defer C.TF_DeleteBuffer(runOptsBuf)

		buf, err := protobuf.Marshal(runOpts)
		if err != nil {
			return nil, err
		}

		runOptsBuf.length = C.size_t(len(buf))
		runOptsBuf.data = C.malloc(runOptsBuf.length)
		if runOptsBuf.data == nil {
			return nil, fmt.Errorf("unable to allocate memory")
		}
		defer C.free(runOptsBuf.data)
		C.memcpy(runOptsBuf.data, unsafe.Pointer(&buf[0]), runOptsBuf.length)
	}

	runMetaData = C.TF_NewBuffer()
  defer C.TF_DeleteBuffer(runMetaData)
  
	C.TF_SessionRun(s.c, runOptsBuf,
		ptrOutput(c.feeds), ptrTensor(c.feedTensors), C.int(len(feeds)),
		ptrOutput(c.fetches), ptrTensor(c.fetchTensors), C.int(len(fetches)),
		ptrOperation(c.targets), C.int(len(targets)),
		runMetaData, status.c)

	// Make sure GC won't harvest input tensors until SessionRun() is finished
	runtime.KeepAlive(feeds)

	if err := status.Err(); err != nil {
		return nil, err
	}
	return c.toGo(), nil
}

// Close a session. This contacts any other processes associated with this
// session, if applicable. Blocks until all previous calls to Run have returned.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.wg.Wait()
	if s.c == nil {
		return nil
	}
	status := newStatus()
	C.TF_CloseSession(s.c, status.c)
	if err := status.Err(); err != nil {
		return err
	}
	C.TF_DeleteSession(s.c, status.c)
	s.c = nil
	return status.Err()
}

// SessionOptions contains configuration information for a session.
type SessionOptions struct {
	// Target indicates the TensorFlow runtime to connect to.
	//
	// If 'target' is empty or unspecified, the local TensorFlow runtime
	// implementation will be used.  Otherwise, the TensorFlow engine
	// defined by 'target' will be used to perform all computations.
	//
	// "target" can be either a single entry or a comma separated list
	// of entries. Each entry is a resolvable address of one of the
	// following formats:
	//   local
	//   ip:port
	//   host:port
	//   ... other system-specific formats to identify tasks and jobs ...
	//
	// NOTE: at the moment 'local' maps to an in-process service-based
	// runtime.
	//
	// Upon creation, a single session affines itself to one of the
	// remote processes, with possible load balancing choices when the
	// "target" resolves to a list of possible processes.
	//
	// If the session disconnects from the remote process during its
	// lifetime, session calls may fail immediately.
	Target string

	// Config is a binary-serialized representation of the
	// tensorflow.ConfigProto protocol message
	// (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto).
	Config []byte
}

// c converts the SessionOptions to the C API's TF_SessionOptions. Callers must
// deallocate by calling the returned done() closure.
func (o *SessionOptions) c() (ret *C.TF_SessionOptions, done func(), err error) {
	opt := C.TF_NewSessionOptions()
	if o == nil {
		return opt, func() { C.TF_DeleteSessionOptions(opt) }, nil
	}
	t := C.CString(o.Target)
	C.TF_SetTarget(opt, t)
	C.free(unsafe.Pointer(t))

	var cConfig unsafe.Pointer
	if sz := len(o.Config); sz > 0 {
		status := newStatus()
		// Copying into C-memory is the simplest thing to do in terms
		// of memory safety and cgo rules ("C code may not keep a copy
		// of a Go pointer after the call returns" from
		// https://golang.org/cmd/cgo/#hdr-Passing_pointers).
		cConfig = C.CBytes(o.Config)
		C.TF_SetConfig(opt, cConfig, C.size_t(sz), status.c)
		if err := status.Err(); err != nil {
			C.TF_DeleteSessionOptions(opt)
			return nil, func() {}, fmt.Errorf("invalid SessionOptions.Config: %v", err)
		}
	}
	return opt, func() {
		C.TF_DeleteSessionOptions(opt)
		C.free(cConfig)
	}, nil
}

// cRunArgs translates the arguments to Session.Run and PartialRun.Run into
// values suitable for C library calls.
type cRunArgs struct {
	feeds        []C.TF_Output
	feedTensors  []*C.TF_Tensor
	fetches      []C.TF_Output
	fetchTensors []*C.TF_Tensor
	targets      []*C.TF_Operation
}

func newCRunArgs(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) *cRunArgs {
	c := &cRunArgs{
		fetches:      make([]C.TF_Output, len(fetches)),
		fetchTensors: make([]*C.TF_Tensor, len(fetches)),
		targets:      make([]*C.TF_Operation, len(targets)),
	}
	for o, t := range feeds {
    c.feeds = append(c.feeds, outputC(o))
		c.feedTensors = append(c.feedTensors, tensorPtrC(t))
	}
	for i, o := range fetches {
		c.fetches[i] = outputC(o)
	}
	for i, t := range targets {
		c.targets[i] = operationPtrC(t)
	}
	return c
}

func (c *cRunArgs) toGo() []*Tensor {
	ret := make([]*Tensor, len(c.fetchTensors))
	for i, ct := range c.fetchTensors {
		ret[i] = newTensorFromC(ct)
	}
	return ret
}

func ptrOutput(l []C.TF_Output) *C.TF_Output {
	if len(l) == 0 {
		return nil
	}
	return &l[0]
}

func ptrTensor(l []*C.TF_Tensor) **C.TF_Tensor {
	if len(l) == 0 {
		return nil
	}
	return &l[0]
}

func ptrOperation(l []*C.TF_Operation) **C.TF_Operation {
	if len(l) == 0 {
		return nil
	}
	return &l[0]
}