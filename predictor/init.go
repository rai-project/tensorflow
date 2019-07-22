package predictor

// #cgo LDFLAGS: -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"os"

	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cast"
)

var (
	log                  *logrus.Entry
	disableOptimizations bool
)

func init() {
	disableOptimizations = cast.ToBool(os.Getenv("CARML_TF_DISABLE_OPTIMIZATION")) == true
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "tensorflow/predictor")
	})
}
