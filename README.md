# Agent for Tensorflow Prediction[![Build Status](https://travis-ci.org/rai-project/tensorflow.svg?branch=master)](https://travis-ci.org/rai-project/tensorflow)

# Install tensorflow 1.12.0 with CUDA 10.0

Build from source -> build the pip package -> GPU support -> bazel build -> ERROR: Config value cuda is not defined in any .rc file
https://github.com/tensorflow/tensorflow/issues/23401

Need to the directory that contains `libtensorflow_framework.so` and `libtensorflow.so` into `$PATH`.
