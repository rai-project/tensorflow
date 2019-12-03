#!/usr/bin/env bash

declare -a array1=(
  MLPerf_ResNet50_v1.5
  VGG16 VGG19
  MLPerf_Mobilenet_v1
  ResNet_v1_50 ResNet_v1_101 ResNet_v1_152 ResNet_v2_50 ResNet_v2_101 ResNet_v2_152
  Inception_ResNet_v2 Inception_v1 Inception_v2 Inception_v3 Inception_v4
  AI_Matrix_ResNet152 AI_Matrix_Densenet121 AI_Matrix_GoogleNet AI_Matrix_ResNet50
  BVLC_AlexNet_Caffe BVLC_GoogLeNet_Caffe
  MobileNet_v1_0.5_128
  MobileNet_v1_0.5_160
  MobileNet_v1_0.5_192 MobileNet_v1_0.5_224
  MobileNet_v1_0.25_128 MobileNet_v1_0.25_160 MobileNet_v1_0.25_192 MobileNet_v1_0.25_224
  MobileNet_v1_0.75_128 MobileNet_v1_0.75_160 MobileNet_v1_0.75_192 MobileNet_v1_0.75_224
  MobileNet_v1_1.0_128 MobileNet_v1_1.0_160 MobileNet_v1_1.0_192
  MobileNet_v1_1.0_224

)

declare -a array2=(
  # Faster_RCNN_Inception_v2_COCO Faster_RCNN_NAS_COCO Faster_RCNN_ResNet101_COCO
  Faster_RCNN_ResNet50_COCO
  MLPerf_SSD_MobileNet_v1_300x300
  MLPerf_SSD_ResNet34_1200x1200
  Mask_RCNN_ResNet50_v2_Atrous_COCO
  Mask_RCNN_Inception_v2_COCO
  Mask_RCNN_Inception_ResNet_v2_Atrous_COCO Mask_RCNN_ResNet101_v2_Atrous_COCO
  SSD_Inception_v2_COCO
  SSD_MobileNet_v1_COCO
  SSD_MobileNet_v2_COCO
  SSD_MobileNet_v1_FPN_Shared_Box_Predictor_640x640_COCO14_Sync SSD_MobileNet_v1_PPN_Shared_Box_Predictor_300x300_COCO14_Sync
)

declare -a array3=(
  DeepLabv3_Xception_65_PASCAL_VOC_Train_Val DeepLabv3_MobileNet_v2_PASCAL_VOC_Train_Val DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Val
  SRGAN
)

# for i in "${array3[@]}"; do
#   echo $i
#   ./gpu_eval_ab.sh localhost 2 $i
#   for ((b = 2; b <= 2; b *= 2)); do
#     ./eurosys_gpu.sh localhost $b $i
#     ./gpu_analysis.sh localhost $b $i
#   done
# done

# for i in "${array2[@]}"; do
#   echo $i
#   ./gpu_eval_ab.sh localhost 16 $i
#   for ((b = 2; b <= 16; b *= 2)); do
#     ./eurosys_gpu.sh localhost $b $i
#     ./gpu_analysis.sh localhost $b $i
#   done
# done

# for i in "${array1[@]}"; do
#   echo $i
#   ./gpu_eval_ab.sh localhost 256 $i
#   for ((b = 2; b <= 256; b *= 2)); do
#     ./eurosys_gpu.sh localhost $b $i
#     ./gpu_analysis.sh localhost $b $i
#   done
# done

for i in "${array1[@]}"; do
  echo $i "on gpu"
  ./eurosys_gpu.sh localhost 1 $i
done

for i in "${array2[@]}"; do
  echo $i "on gpu"
  ./eurosys_gpu.sh localhost 1 $i
done

for i in "${array3[@]}"; do
  echo $i "on gpu"
  ./eurosys_gpu.sh localhost 1 $i
done

for i in "${array1[@]}"; do
  echo $i "on cpu"
  ./eurosys_cpu.sh localhost 1 $i
done

for i in "${array2[@]}"; do
  echo $i "on cpu"
  ./eurosys_cpu.sh localhost 1 $i
done

for i in "${array3[@]}"; do
  echo $i "on cpu"
  ./eurosys_cpu.sh localhost 1 $i
done
