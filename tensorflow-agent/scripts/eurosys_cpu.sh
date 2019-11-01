#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=$3
NUMPREDS=200
OUTPUTFOLDER=output_cpu
DATABASE_NAME=eurosys_cpu
GPU_DEVICE_ID=0

cd ..

if [ ! -d $OUTPUTFOLDER ]; then
  mkdir $OUTPUTFOLDER
fi

if [ -f tensorflow-agent ]; then
  rm tensorflow-agent
fi

go build -tags=nolibjpeg

export TF_CUDNN_USE_AUTOTUNE=0
export CARML_TF_DISABLE_OPTIMIZATION=0
export CUDA_LAUNCH_BLOCKING=0

for ((b = 1; b <= $BATCHSIZE; b *= 2)); do
  echo MODEL_TRACE with batch size $BATCHSIZE
  ./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$(($NUMPREDS * $b)) --database_address=$DATABASE_ADDRESS --publish --batch_size=$b \
    --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID
done
