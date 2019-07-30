#!/usr/bin/env bash

declare -a array=(MLPerf_ResNet50_v1.5 MLPerf_SSD_MobileNet_v1_300x300
  MLPerf_SSD_ResNet34_1200x1200 MLPerf_Mobilenet_v1
  AI_Matrix_ResNet152 AI_Matrix_Densenet121 AI_Matrix_GoogleNet AI_Matrix_ResNet50
  Faster_RCNN_ResNet50_COCO Mask_RCNN_ResNet50_v2_Atrous_COCO
  BVLC_AlexNet_Caffe BVLC_GoogLeNet_Caffe
  DeepLabv3_PASCAL_VOC_Train_Val DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Val
  Faster_RCNN_Inception_v2_COCO Faster_RCNN_NAS_COCO Faster_RCNN_ResNet101_COCO
  Inception_5h Inception_ResNet_v2 Inception_v1 Inception_v2 Inception_v3 Inception_v4
  Mask_RCNN_Inception_v2_COCO Mask_RCNN_Inception_ResNet_v2_Atrous_COCO Mask_RCNN_ResNet101_v2_Atrous_COCO
  MobileNet_v1_0.5_128 MobileNet_v1_0.5_160 MobileNet_v1_0.5_192 MobileNet_v1_0.5_224
  MobileNet_v1_0.25_128 MobileNet_v1_0.25_160 MobileNet_v1_0.25_192 MobileNet_v1_0.25_224
  MobileNet_v1_0.75_128 MobileNet_v1_0.75_160 MobileNet_v1_0.75_192 MobileNet_v1_0.75_224
  MobileNet_v1_1.0_128 MobileNet_v1_1.0_160 MobileNet_v1_1.0_192 MobileNet_v1_1.0_224
  ResNet_v1_50 ResNet_v1_101 ResNet_v1_152 ResNet_v2_50 ResNet_v2_101 ResNet_v2_152
  SRGAN
  SSD_Inception_v2_COCO SSD_MobileNet_v1_COCO SSD_MobileNet_v2_COCO
  SSD_MobileNet_v1_FPN_Shared_Box_Predictor_640x640_COCO14_Sync SSD_MobileNet_v1_PPN_Shared_Box_Predictor_300x300_COCO14_Sync
  VGG16 VGG19
)

for i in "${array[@]}"; do
  echo $i
  ./cpu_eval_ab.sh localhost 16 $i
  ./cpu_eval_fb.sh localhost 1 $i
done

for i in "${array[@]}"; do
  echo $i
  ./cpu_analysis.sh localhost 1 $i
done

