# Agent for Tensorflow Prediction

[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/tensorflow)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=14)
[![Build Status](https://travis-ci.org/rai-project/tensorflow.svg?branch=master)](https://travis-ci.org/rai-project/tensorflow)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-caffe)](https://goreportcard.com/report/github.com/rai-project/go-caffe)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/go-caffe:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:amd64-gpu-latest 'Get your own version badge on microbadger.com')

## Install tensorflow 1.12.0 with CUDA 10.0

Build from source -> build the pip package -> GPU support -> bazel build -> ERROR: Config value cuda is not defined in any .rc file
https://github.com/tensorflow/tensorflow/issues/23401

Need to put the directory that contains `libtensorflow_framework.so` and `libtensorflow.so` into `$PATH`.
