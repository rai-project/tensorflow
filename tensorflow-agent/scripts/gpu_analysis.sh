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

echo "Start to run layer analysis"

./tensorflow-agent evaluation layer info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --format=csv,table --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer"

./tensorflow-agent evaluation layer duration --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_duration.html"

./tensorflow-agent evaluation layer duration --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --box_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_duration.html"

./tensorflow-agent evaluation layer memory --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_memory.html"

./tensorflow-agent evaluation layer occurrence --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --pie_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_occurrence.html"

./tensorflow-agent evaluation layer aggre_duration --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --pie_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_aggre_duration.html"

echo "Start to run gpu analysis"

./tensorflow-agent evaluation gpu_kernel info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel"

./tensorflow-agent evaluation gpu_kernel model_aggre info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_model_aggre"

./tensorflow-agent evaluation gpu_kernel layer_aggre info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_layer_aggre"

./tensorflow-agent evaluation gpu_kernel layer_flops info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_layer_flops.html"

./tensorflow-agent evaluation gpu_kernel layer_dram_read info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_layer_dram_read.html"

./tensorflow-agent evaluation layer_dram_write info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --bar_plot --plot_path="$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_layer_dram_write.html"