package tensorflow

import (
	"io"
	"io/ioutil"
)

func FromCheckpoint(r io.Reader) (*MetaGraphDef, error) {
	var m *MetaGraphDef
	bts, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	err = m.Unmarshal(bts)
	if err != nil {
		return nil, err
	}
	return m, nil
}
