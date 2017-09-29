package tensorflow

import (
	"io"
	"io/ioutil"
)

func FromCheckpoint(r io.Reader) (*GraphDef, error) {

	bts, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	m := new(GraphDef)
	err = m.Unmarshal(bts)
	if err != nil {
		return nil, err
	}
	return m, nil
}
