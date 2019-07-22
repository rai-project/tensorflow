#!/usr/bin/env bash

DATABASE_ADDRESS=$1
BATCHSIZE=$2
MODELNAME=ResNet_v1_50
NUMPREDS=3
DUPLICATE_INPUT=$(($NUMPREDS * $BATCHSIZE))
OUTPUTFOLDER=output
DATABASE_NAME=carml

cd ..

if [ ! -d $OUTPUTFOLDER ]; then
  mkdir $OUTPUTFOLDER
fi

if [ -f tensorflow-agent ]; then
  rm tensorflow-agent
fi
go build -tags=nolibjpeg

# run model trace to get acurate model latency and throughput
for ((b = 1; b <= $BATCHSIZE; b *= 2)); do
  ./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true --batch_size=$b \
    --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME
done

# run the model analysis
./tensorflow-agent evaluation model info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --sort_output --format=csv,table --output="$OUTPUTFOLDER/$MODELNAME/model"

./tensorflow-agent evaluation model info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/model.html"
