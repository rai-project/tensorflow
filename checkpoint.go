package tensorflow

import (
	"io"
	"io/ioutil"

	"github.com/golang/protobuf/proto"
)

func FromCheckpoint(r io.Reader) (*GraphDef, error) {

	bts, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	m := new(GraphDef)
	err = proto.Unmarshal(bts, m)
	if err != nil {
		return nil, err
	}
	return m, nil
}
