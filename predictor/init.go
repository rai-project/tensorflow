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
  enableOptimizations bool 
)

func init() {
	enableOptimizations = cast.ToBool(os.Getenv("CARML_TF_ENABLE_OPTIMIZATION")) == true
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "tensorflow/predictor")
	})
}
