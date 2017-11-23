package main

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry = logger.New().WithField("pkg", "tensorflow/scratch")
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "tensorflow/scratch")
	})
}
