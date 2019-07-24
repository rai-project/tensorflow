#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=MLPerf_ResNet50_v1.5
NUMPREDS=5
OUTPUTFOLDER=output
DATABASE_NAME=test

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

# run model trace to get acurate model latency and throughput
for ((b = 1; b <= $BATCHSIZE; b *= 2)); do
  ./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$(($NUMPREDS * $b)) --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true --batch_size=$b \
    --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME
done

# run the model analysis
./tensorflow-agent evaluation model info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --sort_output --format=csv,table --output="$OUTPUTFOLDER/$MODELNAME/model"

./tensorflow-agent evaluation model info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/model.html"
