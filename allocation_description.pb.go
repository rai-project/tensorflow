// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: allocation_description.proto

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

type AllocationDescription struct {
	// Total number of bytes requested
	RequestedBytes int64 `protobuf:"varint,1,opt,name=requested_bytes,json=requestedBytes,proto3" json:"requested_bytes,omitempty"`
	// Total number of bytes allocated if known
	AllocatedBytes int64 `protobuf:"varint,2,opt,name=allocated_bytes,json=allocatedBytes,proto3" json:"allocated_bytes,omitempty"`
	// Name of the allocator used
	AllocatorName string `protobuf:"bytes,3,opt,name=allocator_name,json=allocatorName,proto3" json:"allocator_name,omitempty"`
	// Identifier of the allocated buffer if known
	AllocationId int64 `protobuf:"varint,4,opt,name=allocation_id,json=allocationId,proto3" json:"allocation_id,omitempty"`
	// Set if this tensor only has one remaining reference
	HasSingleReference bool `protobuf:"varint,5,opt,name=has_single_reference,json=hasSingleReference,proto3" json:"has_single_reference,omitempty"`
	// Address of the allocation.
	Ptr uint64 `protobuf:"varint,6,opt,name=ptr,proto3" json:"ptr,omitempty"`
}

func (m *AllocationDescription) Reset()         { *m = AllocationDescription{} }
func (m *AllocationDescription) String() string { return proto.CompactTextString(m) }
func (*AllocationDescription) ProtoMessage()    {}
func (*AllocationDescription) Descriptor() ([]byte, []int) {
	return fileDescriptor_eebde754afd94ec1, []int{0}
}
func (m *AllocationDescription) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *AllocationDescription) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_AllocationDescription.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *AllocationDescription) XXX_Merge(src proto.Message) {
	xxx_messageInfo_AllocationDescription.Merge(m, src)
}
func (m *AllocationDescription) XXX_Size() int {
	return m.Size()
}
func (m *AllocationDescription) XXX_DiscardUnknown() {
	xxx_messageInfo_AllocationDescription.DiscardUnknown(m)
}

var xxx_messageInfo_AllocationDescription proto.InternalMessageInfo

func (m *AllocationDescription) GetRequestedBytes() int64 {
	if m != nil {
		return m.RequestedBytes
	}
	return 0
}

func (m *AllocationDescription) GetAllocatedBytes() int64 {
	if m != nil {
		return m.AllocatedBytes
	}
	return 0
}

func (m *AllocationDescription) GetAllocatorName() string {
	if m != nil {
		return m.AllocatorName
	}
	return ""
}

func (m *AllocationDescription) GetAllocationId() int64 {
	if m != nil {
		return m.AllocationId
	}
	return 0
}

func (m *AllocationDescription) GetHasSingleReference() bool {
	if m != nil {
		return m.HasSingleReference
	}
	return false
}

func (m *AllocationDescription) GetPtr() uint64 {
	if m != nil {
		return m.Ptr
	}
	return 0
}

func init() {
	proto.RegisterType((*AllocationDescription)(nil), "tensorflow.AllocationDescription")
}

func init() { proto.RegisterFile("allocation_description.proto", fileDescriptor_eebde754afd94ec1) }

var fileDescriptor_eebde754afd94ec1 = []byte{
	// 302 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x6c, 0x91, 0xc1, 0x4a, 0xc3, 0x40,
	0x10, 0x86, 0x3b, 0xb6, 0x16, 0x5d, 0xac, 0xca, 0xa2, 0xb0, 0xa0, 0x2c, 0x51, 0x11, 0x73, 0x31,
	0x15, 0x7c, 0x02, 0x8b, 0x17, 0x2f, 0x52, 0xe2, 0xcd, 0x4b, 0xd8, 0x26, 0xd3, 0x76, 0x35, 0xc9,
	0xc6, 0xd9, 0x2d, 0xc5, 0xb7, 0xf0, 0xb1, 0x3c, 0xf6, 0xe8, 0x51, 0xda, 0x97, 0xd0, 0x9b, 0xa4,
	0xb5, 0x89, 0x07, 0x6f, 0xc3, 0x37, 0x1f, 0xff, 0x2e, 0xff, 0xb0, 0x63, 0x95, 0xa6, 0x26, 0x56,
	0x4e, 0x9b, 0x3c, 0x4a, 0xd0, 0xc6, 0xa4, 0x8b, 0x72, 0x0e, 0x0a, 0x32, 0xce, 0x70, 0xe6, 0x30,
	0xb7, 0x86, 0x86, 0xa9, 0x99, 0x9e, 0x7e, 0x03, 0x3b, 0xbc, 0xa9, 0xe4, 0xdb, 0xda, 0xe5, 0x17,
	0x6c, 0x8f, 0xf0, 0x65, 0x82, 0xd6, 0x61, 0x12, 0x0d, 0x5e, 0x1d, 0x5a, 0x01, 0x1e, 0xf8, 0xcd,
	0x70, 0xb7, 0xc2, 0xbd, 0x92, 0x96, 0xe2, 0xef, 0x73, 0x95, 0xb8, 0xb1, 0x12, 0x2b, 0xbc, 0x12,
	0xcf, 0xd9, 0x9a, 0x18, 0x8a, 0x72, 0x95, 0xa1, 0x68, 0x7a, 0xe0, 0x6f, 0x87, 0x9d, 0x8a, 0xde,
	0xab, 0x0c, 0xf9, 0x19, 0xeb, 0xfc, 0xf9, 0xbe, 0x4e, 0x44, 0x6b, 0x99, 0xb6, 0x53, 0xc3, 0xbb,
	0x84, 0x5f, 0xb1, 0x83, 0xb1, 0xb2, 0x91, 0xd5, 0xf9, 0x28, 0xc5, 0x88, 0x70, 0x88, 0x84, 0x79,
	0x8c, 0x62, 0xd3, 0x03, 0x7f, 0x2b, 0xe4, 0x63, 0x65, 0x1f, 0x96, 0xab, 0x70, 0xbd, 0xe1, 0xfb,
	0xac, 0x59, 0x38, 0x12, 0x6d, 0x0f, 0xfc, 0x56, 0x58, 0x8e, 0xbd, 0xe9, 0xfb, 0x5c, 0xc2, 0x6c,
	0x2e, 0xe1, 0x73, 0x2e, 0xe1, 0x6d, 0x21, 0x1b, 0xb3, 0x85, 0x6c, 0x7c, 0x2c, 0x64, 0x83, 0x09,
	0x43, 0xa3, 0xa0, 0x6e, 0x29, 0x18, 0x92, 0xca, 0x70, 0x6a, 0xe8, 0xb9, 0x77, 0xf4, 0x6f, 0x59,
	0xfd, 0xb2, 0x57, 0xdb, 0x87, 0xc7, 0x93, 0x91, 0x76, 0xe3, 0xc9, 0x20, 0x88, 0x4d, 0xd6, 0x25,
	0xa5, 0x2f, 0x0b, 0x32, 0x4f, 0x18, 0xbb, 0x6e, 0x9d, 0xf5, 0x05, 0x30, 0x68, 0x2f, 0xef, 0x70,
	0xfd, 0x13, 0x00, 0x00, 0xff, 0xff, 0x35, 0x39, 0xd0, 0xc7, 0xa7, 0x01, 0x00, 0x00,
}

func (m *AllocationDescription) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *AllocationDescription) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.RequestedBytes != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintAllocationDescription(dAtA, i, uint64(m.RequestedBytes))
	}
	if m.AllocatedBytes != 0 {
		dAtA[i] = 0x10
		i++
		i = encodeVarintAllocationDescription(dAtA, i, uint64(m.AllocatedBytes))
	}
	if len(m.AllocatorName) > 0 {
		dAtA[i] = 0x1a
		i++
		i = encodeVarintAllocationDescription(dAtA, i, uint64(len(m.AllocatorName)))
		i += copy(dAtA[i:], m.AllocatorName)
	}
	if m.AllocationId != 0 {
		dAtA[i] = 0x20
		i++
		i = encodeVarintAllocationDescription(dAtA, i, uint64(m.AllocationId))
	}
	if m.HasSingleReference {
		dAtA[i] = 0x28
		i++
		if m.HasSingleReference {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i++
	}
	if m.Ptr != 0 {
		dAtA[i] = 0x30
		i++
		i = encodeVarintAllocationDescription(dAtA, i, uint64(m.Ptr))
	}
	return i, nil
}

func encodeVarintAllocationDescription(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *AllocationDescription) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.RequestedBytes != 0 {
		n += 1 + sovAllocationDescription(uint64(m.RequestedBytes))
	}
	if m.AllocatedBytes != 0 {
		n += 1 + sovAllocationDescription(uint64(m.AllocatedBytes))
	}
	l = len(m.AllocatorName)
	if l > 0 {
		n += 1 + l + sovAllocationDescription(uint64(l))
	}
	if m.AllocationId != 0 {
		n += 1 + sovAllocationDescription(uint64(m.AllocationId))
	}
	if m.HasSingleReference {
		n += 2
	}
	if m.Ptr != 0 {
		n += 1 + sovAllocationDescription(uint64(m.Ptr))
	}
	return n
}

func sovAllocationDescription(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozAllocationDescription(x uint64) (n int) {
	return sovAllocationDescription(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *AllocationDescription) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowAllocationDescription
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
			return fmt.Errorf("proto: AllocationDescription: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: AllocationDescription: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RequestedBytes", wireType)
			}
			m.RequestedBytes = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAllocationDescription
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.RequestedBytes |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field AllocatedBytes", wireType)
			}
			m.AllocatedBytes = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAllocationDescription
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.AllocatedBytes |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field AllocatorName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAllocationDescription
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
				return ErrInvalidLengthAllocationDescription
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthAllocationDescription
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.AllocatorName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field AllocationId", wireType)
			}
			m.AllocationId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAllocationDescription
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.AllocationId |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field HasSingleReference", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAllocationDescription
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.HasSingleReference = bool(v != 0)
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Ptr", wireType)
			}
			m.Ptr = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAllocationDescription
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Ptr |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipAllocationDescription(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthAllocationDescription
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthAllocationDescription
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
func skipAllocationDescription(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowAllocationDescription
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
					return 0, ErrIntOverflowAllocationDescription
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
					return 0, ErrIntOverflowAllocationDescription
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
				return 0, ErrInvalidLengthAllocationDescription
			}
			iNdEx += length
			if iNdEx < 0 {
				return 0, ErrInvalidLengthAllocationDescription
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowAllocationDescription
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
				next, err := skipAllocationDescription(dAtA[start:])
				if err != nil {
					return 0, err
				}
				iNdEx = start + next
				if iNdEx < 0 {
					return 0, ErrInvalidLengthAllocationDescription
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
	ErrInvalidLengthAllocationDescription = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowAllocationDescription   = fmt.Errorf("proto: integer overflow")
)
