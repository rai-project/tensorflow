#!/usr/bin/env bash

DATABASE_ADDRESS=$1
MODELNAME=ResNet_v1_50
BATCHSIZE=2
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
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME

# run framework trace to get acurate layer latency
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=FRAMEWORK_TRACE --database_name=$DATABASE_NAME

run system library trace to get acurate layer latency
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --database_name=$DATABASE_NAME

# run system library trace with gpu metrics to get metrics of each cuda kernel
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=flop_count_sp --database_name=$DATABASE_NAME

./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=dram_read_bytes,dram_write_bytes --database_name=$DATABASE_NAME

# run the analysis
./tensorflow-agent evaluation model info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --format=csv --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/model.csv"

./tensorflow-agent evaluation layer info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --format=csv --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer.csv"

./tensorflow-agent evaluation layer latency --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_latency.csv"

./tensorflow-agent evaluation layer duration --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --pie_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_duration.html"

./tensorflow-agent evaluation layer occurrence --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --pie_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_occurrence.html"

./tensorflow-agent evaluation layer memory --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_memory.html"

./tensorflow-agent evaluation cuda_kernel --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/cuda_kernel.csv"
