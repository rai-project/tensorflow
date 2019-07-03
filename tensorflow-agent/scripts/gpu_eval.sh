#!/usr/bin/env bash

DATABASE_ADDRESS=$1
MODELNAME=ResNet_v1_50
BATCHSIZE=1
DUPLICATE_INPUT=5
OUTPUTFOLDER=output

cd ..

if [ ! -d $OUTPUTFOLDER ]; then
  mkdir $OUTPUTFOLDER
fi

go build

# run model trace to get acurate model latency and throughput
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=MODEL_TRACE

# run framework trace to get acurate layer latency
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=FRAMEWORK_TRACE

# run system library trace to get acurate layer latency
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE

# run system library trace with gpu metrics to get metrics of each cuda kernel
./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=flop_count_sp

./tensorflow-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu --disable_autotune=true \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=dram_read_bytes,dram_write_bytes

# run the analysis

# ./tensorflow-agent evaluation latency --database_address=$DATABASE_ADDRESS --model_name=$MODELNAME --batch_size=$BATCHSIZE --format=csv --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/latency.csv"

# ./tensorflow-agent evaluation cuda_kernel --database_address=$DATABASE_ADDRESS --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/cuda_kernel.csv"

# ./tensorflow-agent evaluation layer --database_address=$DATABASE_ADDRESS --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_layer --format=csv --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer.csv"

# ./tensorflow-agent evaluation layer --database_address=$DATABASE_ADDRESS --model_name=$MODELNAME --batch_size=$BATCHSIZE --bar_plot --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/barplot.html"

# ./tensorflow-agent evaluation layer --database_address=$DATABASE_ADDRESS --model_name=$MODELNAME --batch_size=$BATCHSIZE --box_plot --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/boxplot.html"
