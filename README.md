# MLModelScope TensorFlow Agent

[![Build Status](https://travis-ci.org/rai-project/tensorflow.svg?branch=master)](https://travis-ci.org/rai-project/tensorflow)
[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/tensorflow)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=14)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/tensorflow)](https://goreportcard.com/report/github.com/rai-project/tensorflow)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/tensorflow:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/tensorflow:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/tensorflow:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/tensorflow:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/tensorflow:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/tensorflow:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/tensorflow:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/tensorflow:amd64-gpu-latest 'Get your own version badge on microbadger.com')

## Installation

Install go if you have not done so. Please follow [Go Installation](https://docs.mlmodelscope.org/installation/source/golang).

Download and install the MLModelScope TensorFlow Agent:

```
go get -v github.com/rai-project/tensorflow

```

The agent requires The TensorFlow C library and other Go packages.

### Go packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/rai-project/tensorflow
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

Note: The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`


### The TensorFlow C library

The TensorFlow C library is required for the TensorFlow Go package.

If you use TensorFlow Docker Images (e.g. NVIDIA GPU CLOUD (NGC)), skip this step.

You can download pre-built TensorFlow C library from [Install TensorFlow for C](https://www.tensorflow.org/install/lang_c).

Extract the downloaded archive to `/opt/tensorflow/`.

```
tar -C /opt/tensorflow -xzf (downloaded file)
```

Configure the linker environmental variables since the TensorFlow C library is extracted to a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

Linux

```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/tensorflow/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tensorflow/lib
```

macOS

```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/tensorflow/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/tensorflow/lib
```

You can test the installed TensorFlow C library using an [examle C program](https://www.tensorflow.org/install/lang_c#build).

To build the TensorFlow C library from source, refer to [TensorFlow in Go](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/go#building-the-tensorflow-c-library-from-source) .


### libjpeg-turbo

[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) is a JPEG image codec that uses SIMD instructions (MMX, SSE2, AVX2, NEON, AltiVec) to accelerate baseline JPEG compression and decompression. It outperforms libjpeg by a significant amount.

The default is to use libjpeg-turb, to opt-out, use build tag `nolibjpeg`.

To install libjpeg-turbo, refer to [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases).

Linux

```
  export TURBO_VER=2.0.2
  cd /tmp
  wget https://cfhcable.dl.sourceforge.net/project/libjpeg-turbo/${TURBO_VER}/libjpeg-turbo-official_${TURBO_VER}_amd64.deb
  sudo dpkg -i libjpeg-turbo-official_${TURBO_VER}_amd64.deb
```

macOS

```
brew install jpeg-turbo
```

## External services

MLModelScope relies on a few external services.
These services provide tracing, registry, and database servers.

### Installing Docker

Refer to [Install Docker](https://docs.docker.com/install/).

On Ubuntu, an easy way is using

```
curl -fsSL get.docker.com -o get-docker.sh | sudo sh
sudo usermod -aG docker $USER
```

On macOS, [intsall Docker Destop](https://docs.docker.com/docker-for-mac/install/)

### Configuration

You must have a `carml` config file called `.carml_config.yml` under your home directory. An example config file `carml_config.yml.example` is in [github.com/rai-project/MLModelScope](https://github.com/rai-project/MLModelScope) . You can move it to `~/.carml_config.yml`.

The following configuration file can be placed in `$HOME/.carml_config.yml` or can be specified via the `--config="path"` option.

```yaml
app:
  name: carml
  debug: true
  verbose: true
  tempdir: ~/data/carml
registry:
  provider: consul
  endpoints:
    - localhost:8500
  timeout: 20s
  serializer: jsonpb
database:
  provider: mongodb
  endpoints:
    - localhost
tracer:
  enabled: true
  provider: jaeger
  endpoints:
    - localhost:9411
  level: FULL_TRACE
logger:
  hooks:
    - syslog
```

### Starting Trace Server

This service is required.

- On x86 (e.g. intel) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
```

- On ppc64le (e.g. minsky) machines, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) machine by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 carml/jaeger:ppc64le-latest
```

The trace server runs on http://localhost:16686

### Starting Registry Server

This service is not required if using TensorFlow-agent for local evaluation.

- On x86 (e.g. intel) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d consul
```

- On ppc64le (e.g. minsky) machines, start [consul](https://hub.docker.com/_/consul/) by

```
docker run -p 8500:8500 -p 8600:8600 -d carml/consul:ppc64le-latest
```

The registry server runs on http://localhost:8500

### Starting Database Server

This service is not required if not using database to publish evaluation results.

- On x86 (e.g. intel) machines, start [mongodb](https://hub.docker.com/_/mongo/) by

```
docker run -p 27017:27017 --restart always -d mongo:3.0
```

You can also mount the database volume to a local directory using

```
docker run -p 27017:27017 --restart always -d  -v $HOME/data/carml/mongo:/data/db mongo:3.0
```

## Use within TensorFlow Docker Images

Continue if you have

* installed all the dependencies
* downloaded carml_config_example.yml to $HOME as .carml_config.yml
* launched docker external services on the host machine of the docker container you are going to use

, otherwise read above

An example of using NGC TensorFlow docker image: 

```
nvidia-docker run -it --network host \
-v $GOPATH:/workspace/go1.12/global \
-v $GOROOT:/workspace/go1.12_root \
-v ~/.carml_config.yml:/root/.carml_config.yml \
nvcr.io/nvidia/tensorflow:19.05-py2
```

Within the container, set up the environment so that the agent can find the TensorFlow C library.

```
export GOPATH=/workspace/go1.12/global
export GOROOT=/workspace/go1.12_root
export PATH=$GOROOT/bin:$PATH

ln -s /usr/local/lib/tensorflow/libtensorflow_cc.so /usr/local/lib/tensorflow/libtensorflow.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/tensorflow
export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/lib/tensorflow/"

cd $GOPATH/src/github.com/rai-project/tensorflow/tensorflow-agent
```


## Usage

Build the TensorFlow agent with GPU enabled
```
cd $GOPATH/src/github.com/rai-project/tensorflow/tensorflow-agent
go build ./main.go
```

Build the TensorFlow agent without GPU or libjpeg-turbo
```
cd $GOPATH/src/github.com/rai-project/tensorflow/tensorflow-agent
go build -tags="nogpu nolibjpeg" 
```

### Use the agent with the [MLModelScope Web UI](https://github.com/rai-project/mlmodelscope)

```
./tensorflow-agent serve -l -d -v
```

Refer to [TODO] to run the web UI to interact with the agent.

### Use the agent through command line

Run ```./tensorflow-agent -h``` to list the available commands.

Run ```./tensorflow-agent info models``` to list the available models.

Run ```./tensorflow-agent predict``` to evaluate a model. This runs the default evuation. 
```./tensorflow-agent predict -h``` shows the available flags you can set.

An example run is

```
./tensorflow-agent predict urls --trace_level=FRAMEWORK_TRACE --model_name=Inception_v3
```

Refer to [TODO] to run the web UI to interact with the agent.


## Notes on installing TensorFlow from source (ignore this if you are a user)

#### Install Bazel

- [Installing Bazel on Ubuntu](https://docs.bazel.build/versions/master/install-ubuntu.html)

- [Installing Bazel on macOS](https://docs.bazel.build/versions/master/install-os-x.html#install-on-mac-os-x-homebrew)

#### Build

Build TensorFlow 1.13.1 with the following scripts.

```sh
go get -d github.com/tensorflow/tensorflow/tensorflow/go
cd ${GOPATH}/src/github.com/tensorflow/tensorflow
git fetch --all
git checkout v1.13.1
./configure
```

Configure the build and then run 

```
bazel build -c opt //tensorflow:libtensorflow.so
cp ${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow/libtensorflow.so /opt/tensorflow/lib
```

Need to put the directory that contains `libtensorflow_framework.so` and `libtensorflow.so` into `$PATH`.

#### PowerPC

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

#### Issues

- Install tensorflow 1.12 with CUDA 10.0

Currently there's issue using bazel 0.19.1 to build TensorFlow 1.12 with CUDA 10.0.
Build from source -> build the pip package -> GPU support -> bazel build -> ERROR: Config value cuda is not defined in any .rc file https://github.com/tensorflow/tensorflow/issues/23401
