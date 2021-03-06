// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: cluster.proto

package tensorflow

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	io "io"
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

// Defines a single job in a TensorFlow cluster.
type JobDef struct {
	// The name of this job.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// Mapping from task ID to "hostname:port" string.
	//
	// If the `name` field contains "worker", and the `tasks` map contains a
	// mapping from 7 to "example.org:2222", then the device prefix
	// "/job:worker/task:7" will be assigned to "example.org:2222".
	Tasks map[int32]string `protobuf:"bytes,2,rep,name=tasks,proto3" json:"tasks,omitempty" protobuf_key:"varint,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (m *JobDef) Reset()         { *m = JobDef{} }
func (m *JobDef) String() string { return proto.CompactTextString(m) }
func (*JobDef) ProtoMessage()    {}
func (*JobDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_3cfb3b8ec240c376, []int{0}
}
func (m *JobDef) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *JobDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_JobDef.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *JobDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_JobDef.Merge(m, src)
}
func (m *JobDef) XXX_Size() int {
	return m.Size()
}
func (m *JobDef) XXX_DiscardUnknown() {
	xxx_messageInfo_JobDef.DiscardUnknown(m)
}

var xxx_messageInfo_JobDef proto.InternalMessageInfo

func (m *JobDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *JobDef) GetTasks() map[int32]string {
	if m != nil {
		return m.Tasks
	}
	return nil
}

// Defines a TensorFlow cluster as a set of jobs.
type ClusterDef struct {
	// The jobs that comprise the cluster.
	Job []*JobDef `protobuf:"bytes,1,rep,name=job,proto3" json:"job,omitempty"`
}

func (m *ClusterDef) Reset()         { *m = ClusterDef{} }
func (m *ClusterDef) String() string { return proto.CompactTextString(m) }
func (*ClusterDef) ProtoMessage()    {}
func (*ClusterDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_3cfb3b8ec240c376, []int{1}
}
func (m *ClusterDef) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ClusterDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ClusterDef.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ClusterDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ClusterDef.Merge(m, src)
}
func (m *ClusterDef) XXX_Size() int {
	return m.Size()
}
func (m *ClusterDef) XXX_DiscardUnknown() {
	xxx_messageInfo_ClusterDef.DiscardUnknown(m)
}

var xxx_messageInfo_ClusterDef proto.InternalMessageInfo

func (m *ClusterDef) GetJob() []*JobDef {
	if m != nil {
		return m.Job
	}
	return nil
}

func init() {
	proto.RegisterType((*JobDef)(nil), "tensorflow.JobDef")
	proto.RegisterMapType((map[int32]string)(nil), "tensorflow.JobDef.TasksEntry")
	proto.RegisterType((*ClusterDef)(nil), "tensorflow.ClusterDef")
}

func init() { proto.RegisterFile("cluster.proto", fileDescriptor_3cfb3b8ec240c376) }

var fileDescriptor_3cfb3b8ec240c376 = []byte{
	// 269 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xe2, 0x4d, 0xce, 0x29, 0x2d,
	0x2e, 0x49, 0x2d, 0xd2, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0x2a, 0x49, 0xcd, 0x2b, 0xce,
	0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0x57, 0xea, 0x66, 0xe4, 0x62, 0xf3, 0xca, 0x4f, 0x72, 0x49, 0x4d,
	0x13, 0x12, 0xe2, 0x62, 0xc9, 0x4b, 0xcc, 0x4d, 0x95, 0x60, 0x54, 0x60, 0xd4, 0xe0, 0x0c, 0x02,
	0xb3, 0x85, 0x8c, 0xb9, 0x58, 0x4b, 0x12, 0x8b, 0xb3, 0x8b, 0x25, 0x98, 0x14, 0x98, 0x35, 0xb8,
	0x8d, 0x64, 0xf5, 0x10, 0x5a, 0xf5, 0x20, 0xda, 0xf4, 0x42, 0x40, 0xf2, 0xae, 0x79, 0x25, 0x45,
	0x95, 0x41, 0x10, 0xb5, 0x52, 0x16, 0x5c, 0x5c, 0x08, 0x41, 0x21, 0x01, 0x2e, 0xe6, 0xec, 0xd4,
	0x4a, 0xb0, 0xa9, 0xac, 0x41, 0x20, 0xa6, 0x90, 0x08, 0x17, 0x6b, 0x59, 0x62, 0x4e, 0x69, 0xaa,
	0x04, 0x13, 0xd8, 0x26, 0x08, 0xc7, 0x8a, 0xc9, 0x82, 0x51, 0xc9, 0x88, 0x8b, 0xcb, 0x19, 0xe2,
	0x54, 0x90, 0x83, 0x54, 0xb8, 0x98, 0xb3, 0xf2, 0x93, 0x24, 0x18, 0xc1, 0x56, 0x0b, 0x61, 0x5a,
	0x1d, 0x04, 0x92, 0x76, 0xca, 0x3e, 0xf1, 0x48, 0x8e, 0xf1, 0xc2, 0x23, 0x39, 0xc6, 0x07, 0x8f,
	0xe4, 0x18, 0x27, 0x3c, 0x96, 0x63, 0xb8, 0xf0, 0x58, 0x8e, 0xe1, 0xc6, 0x63, 0x39, 0x06, 0x2e,
	0xa9, 0xfc, 0xa2, 0x74, 0x64, 0x5d, 0x29, 0x99, 0xc5, 0x25, 0x45, 0xa5, 0x79, 0x25, 0x99, 0xb9,
	0xa9, 0x4e, 0xbc, 0x50, 0x7b, 0x02, 0x40, 0x21, 0x52, 0x1c, 0xc0, 0x18, 0xa5, 0x98, 0x9e, 0x59,
	0x92, 0x51, 0x9a, 0xa4, 0x97, 0x9c, 0x9f, 0xab, 0x5f, 0x94, 0x98, 0xa9, 0x5b, 0x50, 0x94, 0x9f,
	0x95, 0x9a, 0x5c, 0xa2, 0x8f, 0xd0, 0xff, 0x83, 0x91, 0x31, 0x89, 0x0d, 0x1c, 0x82, 0xc6, 0x80,
	0x00, 0x00, 0x00, 0xff, 0xff, 0x53, 0x25, 0x63, 0x53, 0x52, 0x01, 0x00, 0x00,
}

func (m *JobDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *JobDef) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Name) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintCluster(dAtA, i, uint64(len(m.Name)))
		i += copy(dAtA[i:], m.Name)
	}
	if len(m.Tasks) > 0 {
		for k, _ := range m.Tasks {
			dAtA[i] = 0x12
			i++
			v := m.Tasks[k]
			mapSize := 1 + sovCluster(uint64(k)) + 1 + len(v) + sovCluster(uint64(len(v)))
			i = encodeVarintCluster(dAtA, i, uint64(mapSize))
			dAtA[i] = 0x8
			i++
			i = encodeVarintCluster(dAtA, i, uint64(k))
			dAtA[i] = 0x12
			i++
			i = encodeVarintCluster(dAtA, i, uint64(len(v)))
			i += copy(dAtA[i:], v)
		}
	}
	return i, nil
}

func (m *ClusterDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ClusterDef) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Job) > 0 {
		for _, msg := range m.Job {
			dAtA[i] = 0xa
			i++
			i = encodeVarintCluster(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func encodeVarintCluster(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *JobDef) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovCluster(uint64(l))
	}
	if len(m.Tasks) > 0 {
		for k, v := range m.Tasks {
			_ = k
			_ = v
			mapEntrySize := 1 + sovCluster(uint64(k)) + 1 + len(v) + sovCluster(uint64(len(v)))
			n += mapEntrySize + 1 + sovCluster(uint64(mapEntrySize))
		}
	}
	return n
}

func (m *ClusterDef) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Job) > 0 {
		for _, e := range m.Job {
			l = e.Size()
			n += 1 + l + sovCluster(uint64(l))
		}
	}
	return n
}

func sovCluster(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozCluster(x uint64) (n int) {
	return sovCluster(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *JobDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCluster
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: JobDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: JobDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCluster
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthCluster
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthCluster
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Tasks", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCluster
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthCluster
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthCluster
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Tasks == nil {
				m.Tasks = make(map[int32]string)
			}
			var mapkey int32
			var mapvalue string
			for iNdEx < postIndex {
				entryPreIndex := iNdEx
				var wire uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowCluster
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					wire |= uint64(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				fieldNum := int32(wire >> 3)
				if fieldNum == 1 {
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowCluster
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						mapkey |= int32(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
				} else if fieldNum == 2 {
					var stringLenmapvalue uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowCluster
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						stringLenmapvalue |= uint64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					intStringLenmapvalue := int(stringLenmapvalue)
					if intStringLenmapvalue < 0 {
						return ErrInvalidLengthCluster
					}
					postStringIndexmapvalue := iNdEx + intStringLenmapvalue
					if postStringIndexmapvalue < 0 {
						return ErrInvalidLengthCluster
					}
					if postStringIndexmapvalue > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = string(dAtA[iNdEx:postStringIndexmapvalue])
					iNdEx = postStringIndexmapvalue
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipCluster(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if skippy < 0 {
						return ErrInvalidLengthCluster
					}
					if (iNdEx + skippy) > postIndex {
						return io.ErrUnexpectedEOF
					}
					iNdEx += skippy
				}
			}
			m.Tasks[mapkey] = mapvalue
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipCluster(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCluster
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthCluster
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *ClusterDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCluster
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: ClusterDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ClusterDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Job", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCluster
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthCluster
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthCluster
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Job = append(m.Job, &JobDef{})
			if err := m.Job[len(m.Job)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipCluster(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCluster
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthCluster
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipCluster(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowCluster
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowCluster
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
			return iNdEx, nil
		case 1:
			iNdEx += 8
			return iNdEx, nil
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowCluster
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, ErrInvalidLengthCluster
			}
			iNdEx += length
			if iNdEx < 0 {
				return 0, ErrInvalidLengthCluster
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowCluster
					}
					if iNdEx >= l {
						return 0, io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					innerWire |= (uint64(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				innerWireType := int(innerWire & 0x7)
				if innerWireType == 4 {
					break
				}
				next, err := skipCluster(dAtA[start:])
				if err != nil {
					return 0, err
				}
				iNdEx = start + next
				if iNdEx < 0 {
					return 0, ErrInvalidLengthCluster
				}
			}
			return iNdEx, nil
		case 4:
			return iNdEx, nil
		case 5:
			iNdEx += 4
			return iNdEx, nil
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
	}
	panic("unreachable")
}

var (
	ErrInvalidLengthCluster = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowCluster   = fmt.Errorf("proto: integer overflow")
)
