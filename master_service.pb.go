// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: master_service.proto

package tensorflow

import (
	context "context"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	grpc "google.golang.org/grpc"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion2 // please upgrade the proto package

func init() { proto.RegisterFile("master_service.proto", fileDescriptor_9a501bcba839fe29) }

var fileDescriptor_9a501bcba839fe29 = []byte{
	// 387 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x93, 0xbd, 0x4e, 0xf3, 0x30,
	0x14, 0x86, 0x9b, 0xe1, 0xeb, 0x27, 0x99, 0x56, 0x95, 0x02, 0x03, 0x04, 0xc9, 0xfd, 0x61, 0x26,
	0x95, 0x60, 0x65, 0x6a, 0x61, 0xa3, 0xa8, 0xa4, 0x4c, 0x5d, 0x90, 0x9b, 0x1e, 0x8a, 0x21, 0xb5,
	0x83, 0xed, 0x00, 0x97, 0xc1, 0xfd, 0x70, 0x03, 0x8c, 0x1d, 0x19, 0x51, 0x7b, 0x23, 0x28, 0x4d,
	0xd2, 0xd8, 0x21, 0xed, 0x7a, 0x9e, 0xd7, 0x8f, 0xf2, 0x3a, 0x3e, 0xe8, 0x60, 0x4e, 0xa4, 0x02,
	0x71, 0x2f, 0x41, 0xbc, 0x52, 0x1f, 0xdc, 0x50, 0x70, 0xc5, 0xed, 0x86, 0x02, 0x26, 0xb9, 0x78,
	0x08, 0xf8, 0x9b, 0x3b, 0x13, 0xa1, 0xef, 0xd4, 0x92, 0x58, 0x82, 0xcf, 0x3e, 0xab, 0xa8, 0x3e,
	0x58, 0x0f, 0x46, 0xc9, 0x31, 0xfb, 0x0e, 0xd5, 0xfb, 0x02, 0x88, 0x82, 0x11, 0x48, 0x49, 0x39,
	0xb3, 0x5b, 0xae, 0xa6, 0x30, 0x90, 0x07, 0x2f, 0x11, 0x48, 0xe5, 0xb4, 0x77, 0x24, 0x64, 0xc8,
	0x99, 0x5c, 0x5b, 0xaf, 0xde, 0x15, 0xb0, 0x69, 0xa9, 0xd5, 0x40, 0xa5, 0xd6, 0x42, 0x22, 0xb5,
	0x8e, 0x51, 0x63, 0x48, 0x84, 0xa2, 0x24, 0xf0, 0x22, 0x36, 0x02, 0x15, 0x85, 0x76, 0x47, 0x3f,
	0x55, 0x80, 0x99, 0xf9, 0x64, 0x67, 0x26, 0x75, 0xf7, 0xd0, 0xff, 0x78, 0xa6, 0x20, 0xb4, 0x1d,
	0x3d, 0x9f, 0x0e, 0x33, 0xd7, 0x71, 0x29, 0x4b, 0x1d, 0xb7, 0xa8, 0xd6, 0x0f, 0xb8, 0xdc, 0x5c,
	0x65, 0xd3, 0xb8, 0x28, 0x8d, 0x64, 0xb6, 0xd6, 0xf6, 0x40, 0xaa, 0xbc, 0x41, 0x7b, 0xd7, 0x54,
	0xaa, 0x4b, 0x88, 0x7f, 0x96, 0xb4, 0xb1, 0x7e, 0x40, 0x03, 0x99, 0xb0, 0xb9, 0x95, 0xa7, 0xbe,
	0x0b, 0xf4, 0xcf, 0x03, 0x09, 0xca, 0x3e, 0x34, 0x8a, 0xc4, 0xa3, 0xcc, 0x71, 0x54, 0x42, 0xf2,
	0x82, 0x03, 0xf2, 0x0c, 0x7d, 0x12, 0x04, 0x64, 0x12, 0x80, 0x59, 0x50, 0x27, 0xa5, 0x05, 0xcd,
	0x40, 0x5e, 0xd0, 0x8b, 0xd8, 0xc6, 0x88, 0x0b, 0xf7, 0x5b, 0x14, 0x36, 0xb7, 0xf2, 0xfc, 0x8d,
	0x78, 0x10, 0x00, 0x91, 0xf9, 0x57, 0x76, 0xcc, 0x42, 0x06, 0x2c, 0x7d, 0x23, 0x7f, 0x32, 0x89,
	0xbb, 0xc7, 0xbe, 0x96, 0xd8, 0x5a, 0x2c, 0xb1, 0xf5, 0xb3, 0xc4, 0xd6, 0xc7, 0x0a, 0x57, 0x16,
	0x2b, 0x5c, 0xf9, 0x5e, 0xe1, 0x0a, 0x72, 0xb8, 0x98, 0xe9, 0x86, 0x29, 0x95, 0x4a, 0x44, 0x4c,
	0xd1, 0x39, 0xf4, 0xf6, 0x8d, 0x85, 0x1b, 0xc6, 0x7b, 0x28, 0x87, 0xd6, 0xb8, 0x3d, 0xa3, 0xea,
	0x31, 0x9a, 0xb8, 0x3e, 0x9f, 0x77, 0x05, 0xa1, 0xa7, 0xa1, 0xe0, 0x4f, 0xe0, 0xab, 0x6e, 0x6e,
	0x99, 0x54, 0xd7, 0x4b, 0x7b, 0xfe, 0x1b, 0x00, 0x00, 0xff, 0xff, 0xce, 0xd6, 0x49, 0x4d, 0xeb,
	0x03, 0x00, 0x00,
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// MasterServiceClient is the client API for MasterService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type MasterServiceClient interface {
	// Creates a session.
	CreateSession(ctx context.Context, in *CreateSessionRequest, opts ...grpc.CallOption) (*CreateSessionResponse, error)
	// Extends a session.
	ExtendSession(ctx context.Context, in *ExtendSessionRequest, opts ...grpc.CallOption) (*ExtendSessionResponse, error)
	// Prepares future partial run calls.
	PartialRunSetup(ctx context.Context, in *PartialRunSetupRequest, opts ...grpc.CallOption) (*PartialRunSetupResponse, error)
	// Drives the graph computation.
	RunStep(ctx context.Context, in *RunStepRequest, opts ...grpc.CallOption) (*RunStepResponse, error)
	// Closes a session.
	CloseSession(ctx context.Context, in *CloseSessionRequest, opts ...grpc.CallOption) (*CloseSessionResponse, error)
	// List the devices usable by the master.
	ListDevices(ctx context.Context, in *ListDevicesRequest, opts ...grpc.CallOption) (*ListDevicesResponse, error)
	// Close and abandon all existing sessions.  Ongoing computations
	// will no longer affect fresh ones via the resources in containers listed in
	// the ResetRequest.  See ResetRequest for more details.
	Reset(ctx context.Context, in *ResetRequest, opts ...grpc.CallOption) (*ResetResponse, error)
	// Registers a callable for execution with RunCallable.
	MakeCallable(ctx context.Context, in *MakeCallableRequest, opts ...grpc.CallOption) (*MakeCallableResponse, error)
	// Executes a callable registered with MakeCallable.
	RunCallable(ctx context.Context, in *RunCallableRequest, opts ...grpc.CallOption) (*RunCallableResponse, error)
	// Frees resources associated with a callable registered with MakeCallable.
	ReleaseCallable(ctx context.Context, in *ReleaseCallableRequest, opts ...grpc.CallOption) (*ReleaseCallableResponse, error)
}

type masterServiceClient struct {
	cc *grpc.ClientConn
}

func NewMasterServiceClient(cc *grpc.ClientConn) MasterServiceClient {
	return &masterServiceClient{cc}
}

func (c *masterServiceClient) CreateSession(ctx context.Context, in *CreateSessionRequest, opts ...grpc.CallOption) (*CreateSessionResponse, error) {
	out := new(CreateSessionResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/CreateSession", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) ExtendSession(ctx context.Context, in *ExtendSessionRequest, opts ...grpc.CallOption) (*ExtendSessionResponse, error) {
	out := new(ExtendSessionResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/ExtendSession", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) PartialRunSetup(ctx context.Context, in *PartialRunSetupRequest, opts ...grpc.CallOption) (*PartialRunSetupResponse, error) {
	out := new(PartialRunSetupResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/PartialRunSetup", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) RunStep(ctx context.Context, in *RunStepRequest, opts ...grpc.CallOption) (*RunStepResponse, error) {
	out := new(RunStepResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/RunStep", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) CloseSession(ctx context.Context, in *CloseSessionRequest, opts ...grpc.CallOption) (*CloseSessionResponse, error) {
	out := new(CloseSessionResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/CloseSession", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) ListDevices(ctx context.Context, in *ListDevicesRequest, opts ...grpc.CallOption) (*ListDevicesResponse, error) {
	out := new(ListDevicesResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/ListDevices", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) Reset(ctx context.Context, in *ResetRequest, opts ...grpc.CallOption) (*ResetResponse, error) {
	out := new(ResetResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/Reset", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) MakeCallable(ctx context.Context, in *MakeCallableRequest, opts ...grpc.CallOption) (*MakeCallableResponse, error) {
	out := new(MakeCallableResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/MakeCallable", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) RunCallable(ctx context.Context, in *RunCallableRequest, opts ...grpc.CallOption) (*RunCallableResponse, error) {
	out := new(RunCallableResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/RunCallable", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *masterServiceClient) ReleaseCallable(ctx context.Context, in *ReleaseCallableRequest, opts ...grpc.CallOption) (*ReleaseCallableResponse, error) {
	out := new(ReleaseCallableResponse)
	err := c.cc.Invoke(ctx, "/tensorflow.grpc.MasterService/ReleaseCallable", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// MasterServiceServer is the server API for MasterService service.
type MasterServiceServer interface {
	// Creates a session.
	CreateSession(context.Context, *CreateSessionRequest) (*CreateSessionResponse, error)
	// Extends a session.
	ExtendSession(context.Context, *ExtendSessionRequest) (*ExtendSessionResponse, error)
	// Prepares future partial run calls.
	PartialRunSetup(context.Context, *PartialRunSetupRequest) (*PartialRunSetupResponse, error)
	// Drives the graph computation.
	RunStep(context.Context, *RunStepRequest) (*RunStepResponse, error)
	// Closes a session.
	CloseSession(context.Context, *CloseSessionRequest) (*CloseSessionResponse, error)
	// List the devices usable by the master.
	ListDevices(context.Context, *ListDevicesRequest) (*ListDevicesResponse, error)
	// Close and abandon all existing sessions.  Ongoing computations
	// will no longer affect fresh ones via the resources in containers listed in
	// the ResetRequest.  See ResetRequest for more details.
	Reset(context.Context, *ResetRequest) (*ResetResponse, error)
	// Registers a callable for execution with RunCallable.
	MakeCallable(context.Context, *MakeCallableRequest) (*MakeCallableResponse, error)
	// Executes a callable registered with MakeCallable.
	RunCallable(context.Context, *RunCallableRequest) (*RunCallableResponse, error)
	// Frees resources associated with a callable registered with MakeCallable.
	ReleaseCallable(context.Context, *ReleaseCallableRequest) (*ReleaseCallableResponse, error)
}

func RegisterMasterServiceServer(s *grpc.Server, srv MasterServiceServer) {
	s.RegisterService(&_MasterService_serviceDesc, srv)
}

func _MasterService_CreateSession_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CreateSessionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).CreateSession(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/CreateSession",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).CreateSession(ctx, req.(*CreateSessionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_ExtendSession_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ExtendSessionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).ExtendSession(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/ExtendSession",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).ExtendSession(ctx, req.(*ExtendSessionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_PartialRunSetup_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PartialRunSetupRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).PartialRunSetup(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/PartialRunSetup",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).PartialRunSetup(ctx, req.(*PartialRunSetupRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_RunStep_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RunStepRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).RunStep(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/RunStep",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).RunStep(ctx, req.(*RunStepRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_CloseSession_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CloseSessionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).CloseSession(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/CloseSession",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).CloseSession(ctx, req.(*CloseSessionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_ListDevices_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ListDevicesRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).ListDevices(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/ListDevices",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).ListDevices(ctx, req.(*ListDevicesRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_Reset_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ResetRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).Reset(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/Reset",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).Reset(ctx, req.(*ResetRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_MakeCallable_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(MakeCallableRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).MakeCallable(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/MakeCallable",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).MakeCallable(ctx, req.(*MakeCallableRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_RunCallable_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RunCallableRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).RunCallable(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/RunCallable",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).RunCallable(ctx, req.(*RunCallableRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _MasterService_ReleaseCallable_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ReleaseCallableRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MasterServiceServer).ReleaseCallable(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/tensorflow.grpc.MasterService/ReleaseCallable",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MasterServiceServer).ReleaseCallable(ctx, req.(*ReleaseCallableRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _MasterService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "tensorflow.grpc.MasterService",
	HandlerType: (*MasterServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "CreateSession",
			Handler:    _MasterService_CreateSession_Handler,
		},
		{
			MethodName: "ExtendSession",
			Handler:    _MasterService_ExtendSession_Handler,
		},
		{
			MethodName: "PartialRunSetup",
			Handler:    _MasterService_PartialRunSetup_Handler,
		},
		{
			MethodName: "RunStep",
			Handler:    _MasterService_RunStep_Handler,
		},
		{
			MethodName: "CloseSession",
			Handler:    _MasterService_CloseSession_Handler,
		},
		{
			MethodName: "ListDevices",
			Handler:    _MasterService_ListDevices_Handler,
		},
		{
			MethodName: "Reset",
			Handler:    _MasterService_Reset_Handler,
		},
		{
			MethodName: "MakeCallable",
			Handler:    _MasterService_MakeCallable_Handler,
		},
		{
			MethodName: "RunCallable",
			Handler:    _MasterService_RunCallable_Handler,
		},
		{
			MethodName: "ReleaseCallable",
			Handler:    _MasterService_ReleaseCallable_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "master_service.proto",
}
