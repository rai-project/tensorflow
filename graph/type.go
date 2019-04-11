package graph

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// DataType holds the type for a scalar value.  E.g., one slot in a tensor.
type DataType tf.DataType

const (
	Unknown    DataType = DataType(0)
	Float      DataType = DataType(tf.Float)
	Double     DataType = DataType(tf.Double)
	Int32      DataType = DataType(tf.Int32)
	Uint32     DataType = DataType(tf.Uint32)
	Uint8      DataType = DataType(tf.Uint8)
	Int16      DataType = DataType(tf.Int16)
	Int8       DataType = DataType(tf.Int8)
	String     DataType = DataType(tf.String)
	Complex64  DataType = DataType(tf.Complex64)
	Complex    DataType = DataType(tf.Complex)
	Int64      DataType = DataType(tf.Int64)
	Uint64     DataType = DataType(tf.Uint64)
	Bool       DataType = DataType(tf.Bool)
	Qint8      DataType = DataType(tf.Qint8)
	Quint8     DataType = DataType(tf.Quint8)
	Qint32     DataType = DataType(tf.Qint32)
	Bfloat16   DataType = DataType(tf.Bfloat16)
	Qint16     DataType = DataType(tf.Qint16)
	Quint16    DataType = DataType(tf.Quint16)
	Uint16     DataType = DataType(tf.Uint16)
	Complex128 DataType = DataType(tf.Complex128)
	Half       DataType = DataType(tf.Half)
)

func (d DataType) String() string {
	switch d {
	case Float:
		return "Float"
	case Double:
		return "Double"
	case Int32:
		return "Int32"
	case Uint32:
		return "Uint32"
	case Uint8:
		return "Uint8"
	case Int16:
		return "Int16"
	case Int8:
		return "Int8"
	case String:
		return "String"
	case Complex64:
		return "Complex64"
	case Int64:
		return "Int64"
	case Uint64:
		return "Uint64"
	case Bool:
		return "Bool"
	case Qint8:
		return "Qint8"
	case Quint8:
		return "Quint8"
	case Qint32:
		return "Qint32"
	case Bfloat16:
		return "Bfloat16"
	case Qint16:
		return "Qint16"
	case Quint16:
		return "Quint16"
	case Uint16:
		return "Uint16"
	case Complex128:
		return "Complex128"
	case Half:
		return "Half"
	}
	return "Unknown"
}

func (d DataType) ByteCount() int {
	switch d {
	case Float:
		return 8
	case Double:
		return 16
	case Int32:
		return 8
	case Uint32:
		return 8
	case Uint8:
		return 1
	case Int16:
		return 2
	case Int8:
		return 1
	case String:
		return 1
	case Complex64:
		return 32
	case Int64:
		return 16
	case Uint64:
		return 16
	case Bool:
		return 1
	case Qint8:
		return 1
	case Quint8:
		return 1
	case Qint32:
		return 8
	case Bfloat16:
		return 2
	case Qint16:
		return 2
	case Quint16:
		return 2
	case Uint16:
		return 2
	case Complex128:
		return 64
	case Half:
		return 2
	}
	return 0
}

func GetDataType(s string) DataType {
	switch s {
	case "Float":
		return Float
	case "Double":
		return Double
	case "Int32":
		return Int32
	case "Uint32":
		return Uint32
	case "Uint8":
		return Uint8
	case "Int16":
		return Int16
	case "Int8":
		return Int8
	case "String":
		return String
	case "Complex64":
		return Complex64
	case "Complex":
		return Complex
	case "Int64":
		return Int64
	case "Uint64":
		return Uint64
	case "Bool":
		return Bool
	case "Qint8":
		return Qint8
	case "Quint8":
		return Quint8
	case "Qint32":
		return Qint32
	case "Bfloat16":
		return Bfloat16
	case "Qint16":
		return Qint16
	case "Quint16":
		return Quint16
	case "Uint16":
		return Uint16
	case "Complex128":
		return Complex128
	case "Half":
		return Half
	}
	return Unknown
}
