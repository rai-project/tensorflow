package predictor

// #cgo LDFLAGS: -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry
)


type operation struct {
	c *C.TF_Operation
	// A reference to the Graph to prevent it from
	// being GCed while the Operation is still alive.
	g *Graph
}


func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "tensorflow/predict")
	})
}
