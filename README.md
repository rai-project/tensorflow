# Agent for Tensorflow Prediction

[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/tensorflow)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=14)
[![Build Status](https://travis-ci.org/rai-project/tensorflow.svg?branch=master)](https://travis-ci.org/rai-project/tensorflow)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-caffe)](https://goreportcard.com/report/github.com/rai-project/go-caffe)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/go-caffe:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:amd64-gpu-latest 'Get your own version badge on microbadger.com')

Refer to [TensorFlow in Go](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go#building-the-tensorflow-c-library-from-source) for instructions

### Install Bazel

{{% notice note %}}
Currently there's issue using bazel 0.19.1 to build TensorFlow 1.12 with CUDA 10.0
{{% /notice %}}

- [Installing Bazel on Ubuntu](https://docs.bazel.build/versions/master/install-ubuntu.html)

- [Installing Bazel on macOS](https://docs.bazel.build/versions/master/install-os-x.html#install-on-mac-os-x-homebrew)

### Build

Build TensorFlow 1.12 with the following scripts.

```sh
go get -d github.com/tensorflow/tensorflow/tensorflow/go
cd ${GOPATH}/src/github.com/tensorflow/tensorflow
git fetch --all
git checkout r1.12
./configure
```

For linux with gpu, an example `.tf_configure.bazelrc` is

```
build --action_env PYTHON_BIN_PATH="/usr/bin/python"
build --action_env PYTHON_LIB_PATH="/usr/lib/python3/dist-packages"
build --python_path="/usr/bin/python"
build:ignite --define with_ignite_support=true
build --define with_xla_support=true
build --action_env TF_NEED_OPENCL_SYCL="0"
build --action_env TF_NEED_ROCM="0"
build --action_env TF_NEED_CUDA="1"
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda"
build --action_env TF_CUDA_VERSION="10.0"
build --action_env CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"
build --action_env TF_CUDNN_VERSION="7"
build --action_env TENSORRT_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"
build --action_env TF_TENSORRT_VERSION="5.0.0"
build --action_env NCCL_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"
build --action_env NCCL_HDR_PATH="/usr/include"
build --action_env TF_NCCL_VERSION="2"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="3.5,7.0"
build --action_env LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:/home/abduld/.gvm/pkgsets/go1.11/global/overlay/lib"
build --action_env TF_CUDA_CLANG="0"
build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
build --config=cuda
test --config=cuda
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
build:v2 --define=tf_api_version=2
```

For macos without gpu, an example `.tf_configure.bazelrc` is

```
build --action_env PYTHON_BIN_PATH="/usr/local/opt/python/bin/python3.7"
build --action_env PYTHON_LIB_PATH="/usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages"
build --python_path="/usr/local/opt/python/bin/python3.7"
build:ignite --define with_ignite_support=true
build --define with_xla_support=true
build --action_env TF_NEED_OPENCL_SYCL="0"
build --action_env TF_NEED_ROCM="0"
build --action_env TF_NEED_CUDA="0"
build --action_env TF_DOWNLOAD_CLANG="0"
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
build:v2 --define=tf_api_version=2
```

Then run

```bash
bazel build -c opt //tensorflow:libtensorflow.so
cp ${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow/libtensorflow.so /opt/tensorflow/lib
```

### Setting `LD_LIBRARY_PATH`

Place the following in either your `~/.bashrc` or `~/.zshrc` file

```bash
export LD_LIBRARY_PATH=/opt/tensorflow/lib:$LD_LIBRARY_PATH
```

### PowerPC

For TensorFlow compilation, here are the recommended tensorflow-configure settings:

```
export CC_OPT_FLAGS="-mcpu=power8 -mtune=power8"
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc

ANACONDA_HOME=$(conda info --json | python -c "import sys, json; print json.load(sys.stdin)['default_prefix']")
export PYTHON_BIN_PATH=$ANACONDA_HOME/bin/python
export PYTHON_LIB_PATH=$ANACONDA_HOME/lib/python2.7/site-packages

export USE_DEFAULT_PYTHON_LIB_PATH=0
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=9.0
export CUDA_TOOLKIT_PATH=/usr/local/cuda-9.0
export TF_CUDA_COMPUTE_CAPABILITIES=3.5,3.7,5.2,6.0,7.0
export CUDNN_INSTALL_PATH=/usr/local/cuda-9.0
export TF_CUDNN_VERSION=7
export TF_NEED_GCP=1
export TF_NEED_OPENCL=0
export TF_NEED_HDFS=1
export TF_NEED_JEMALLOC=1
export TF_ENABLE_XLA=1
export TF_CUDA_CLANG=0
export TF_NEED_MKL=0
export TF_NEED_MPI=0
export TF_NEED_VERBS=0
export TF_NEED_GDR=0
export TF_NEED_S3=0
```

## Issues

### Install tensorflow 1.12.0 with CUDA 10.0

Build from source -> build the pip package -> GPU support -> bazel build -> ERROR: Config value cuda is not defined in any .rc file
https://github.com/tensorflow/tensorflow/issues/23401

Need to put the directory that contains `libtensorflow_framework.so` and `libtensorflow.so` into `$PATH`.
