#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=ResNet_v1_50
NUMPREDS=5
DUPLICATE_INPUT=$(($NUMPREDS * $BATCHSIZE))
OUTPUTFOLDER=output
DATABASE_NAME=carml

cd ..

if [ -f tensorflow-agent ]; then
  rm tensorflow-agent
fi
go build -tags=nolibjpeg

export TF_CUDNN_USE_AUTOTUNE=0
export CARML_TF_DISABLE_OPTIMIZATION=0
export CUDA_LAUNCH_BLOCKING=0

./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true --batch_size=$BATCHSIZE \
  --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME

export CUDA_LAUNCH_BLOCKING=1

# run framework trace to get acurate layer latency
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=FRAMEWORK_TRACE --database_name=$DATABASE_NAME

export CUDA_LAUNCH_BLOCKING=0

# run system library trace to get acurate gpu kernel latency
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --database_name=$DATABASE_NAME

# run system library trace with gpu metrics to get metrics of each cuda kernel
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=flop_count_sp --database_name=$DATABASE_NAME

./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=dram_read_bytes,dram_write_bytes --database_name=$DATABASE_NAME
