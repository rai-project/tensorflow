#!/usr/bin/env bash

declare -a array=(
  MLPerf_ResNet50_v1.5
  VGG16 VGG19
  MLPerf_Mobilenet_v1
  ResNet_v1_50 ResNet_v1_101 ResNet_v1_152 ResNet_v2_50 ResNet_v2_101 ResNet_v2_152
  Inception_ResNet_v2 Inception_v1 Inception_v2 Inception_v3 Inception_v4
  AI_Matrix_ResNet152 AI_Matrix_Densenet121 AI_Matrix_GoogleNet AI_Matrix_ResNet50
  BVLC_AlexNet_Caffe BVLC_GoogLeNet_Caffe
  Faster_RCNN_Inception_v2_COCO Faster_RCNN_NAS_COCO Faster_RCNN_ResNet101_COCO
  Faster_RCNN_ResNet50_COCO Mask_RCNN_ResNet50_v2_Atrous_COCO
  Mask_RCNN_Inception_v2_COCO Mask_RCNN_Inception_ResNet_v2_Atrous_COCO Mask_RCNN_ResNet101_v2_Atrous_COCO
  MLPerf_SSD_MobileNet_v1_300x300
  MLPerf_SSD_ResNet34_1200x1200
  SSD_Inception_v2_COCO SSD_MobileNet_v1_COCO SSD_MobileNet_v2_COCO
  SSD_MobileNet_v1_FPN_Shared_Box_Predictor_640x640_COCO14_Sync SSD_MobileNet_v1_PPN_Shared_Box_Predictor_300x300_COCO14_Sync
  MobileNet_v1_0.5_128
  MobileNet_v1_0.5_160 MobileNet_v1_0.5_192 MobileNet_v1_0.5_224
  MobileNet_v1_0.25_128 MobileNet_v1_0.25_160 MobileNet_v1_0.25_192 MobileNet_v1_0.25_224
  MobileNet_v1_0.75_128 MobileNet_v1_0.75_160 MobileNet_v1_0.75_192 MobileNet_v1_0.75_224
  MobileNet_v1_1.0_128 MobileNet_v1_1.0_160 MobileNet_v1_1.0_192
  MobileNet_v1_1.0_224
  DeepLabv3_PASCAL_VOC_Train_Val DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Val
  SRGAN
)

# # # echo Faster_RCNN_Inception_v2_COCO
# ./gpu_eval_fb.sh localhost 4 Faster_RCNN_Inception_v2_COCO
# ./gpu_analysis.sh localhost 4 Faster_RCNN_Inception_v2_COCO

# # # echo Faster_RCNN_NAS_COCO
# ./gpu_eval_fb.sh localhost 4 Faster_RCNN_NAS_COCO
# ./gpu_analysis.sh localhost 4 Faster_RCNN_NAS_COCO

# # # echo Faster_RCNN_ResNet101_COCO
# ./gpu_eval_fb.sh localhost 4 Faster_RCNN_ResNet101_COCO
# ./gpu_analysis.sh localhost 4 Faster_RCNN_ResNet101_COCO

# # # echo Faster_RCNN_ResNet50_COCO
# ./gpu_eval_fb.sh localhost 4 Faster_RCNN_ResNet50_COCO
# ./gpu_analysis.sh localhost 4 Faster_RCNN_ResNet50_COCO

echo SSD_Inception_v2_COCO
./gpu_eval_fb.sh localhost 8 SSD_Inception_v2_COCO
# ./gpu_analysis.sh localhost 8 SSD_Inception_v2_COCO

# echo MLPerf_SSD_MobileNet_v1_300x300
# ./gpu_eval_fb.sh localhost 8 MLPerf_SSD_MobileNet_v1_300x300
# ./gpu_analysis.sh localhost 8 MLPerf_SSD_MobileNet_v1_300x300

# echo SSD_MobileNet_v2_COCO
# ./gpu_eval_fb.sh localhost 8 SSD_MobileNet_v2_COCO
# ./gpu_analysis.sh localhost 8 SSD_MobileNet_v2_COCO

# echo SSD_MobileNet_v1_PPN_Shared_Box_Predictor_300x300_COCO14_Sync
# ./gpu_eval_fb.sh localhost 16 SSD_MobileNet_v1_PPN_Shared_Box_Predictor_300x300_COCO14_Sync
# ./gpu_analysis.sh localhost 16 SSD_MobileNet_v1_PPN_Shared_Box_Predictor_300x300_COCO14_Sync

# echo Mask_RCNN_Inception_ResNet_v2_Atrous_COCO
# ./gpu_eval_fb.sh localhost 4 Mask_RCNN_Inception_ResNet_v2_Atrous_COCO
# ./gpu_analysis.sh localhost 4 Mask_RCNN_Inception_ResNet_v2_Atrous_COCO

# echo Mask_RCNN_ResNet101_v2_Atrous_COCO
# ./gpu_eval_fb.sh localhost 2 Mask_RCNN_ResNet101_v2_Atrous_COCO
# ./gpu_analysis.sh localhost 2 Mask_RCNN_ResNet101_v2_Atrous_COCO

# echo Mask_RCNN_ResNet50_v2_Atrous_COCO
# ./gpu_eval_fb.sh localhost 2 Mask_RCNN_ResNet50_v2_Atrous_COCO
# ./gpu_analysis.sh localhost 2 Mask_RCNN_ResNet50_v2_Atrous_COCO

# echo Mask_RCNN_Inception_v2_COCO
# ./gpu_eval_fb.sh localhost 4 Mask_RCNN_Inception_v2_COCO
# ./gpu_analysis.sh localhost 4 Mask_RCNN_Inception_v2_COCO

# echo DeepLabv3_MobileNet_v2_PASCAL_VOC_Train_Val
# ./gpu_eval_fb.sh localhost 1 DeepLabv3_MobileNet_v2_PASCAL_VOC_Train_Val
# ./gpu_analysis.sh localhost 1 DeepLabv3_MobileNet_v2_PASCAL_VOC_Train_Val

# echo DeepLabv3_PASCAL_VOC_Train_Val
# ./gpu_eval_fb.sh localhost 1 DeepLabv3_PASCAL_VOC_Train_Val
# ./gpu_analysis.sh localhost 1 DeepLabv3_PASCAL_VOC_Train_Val

# echo DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Val
# ./gpu_eval_fb.sh localhost 1 DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Val
# ./gpu_analysis.sh localhost 1 DeepLabv3_MobileNet_v2_DM_05_PASCAL_VOC_Train_Val

# # echo SSD_MobileNet_v1_FPN_Shared_Box_Predictor_640x640_COCO14_Sync
# ./gpu_eval_fb.sh localhost 8 SSD_MobileNet_v1_FPN_Shared_Box_Predictor_640x640_COCO14_Sync
# ./gpu_analysis.sh localhost 8 SSD_MobileNet_v1_FPN_Shared_Box_Predictor_640x640_COCO14_Sync

# echo Inception_ResNet_v2
# ./gpu_eval_fb.sh localhost 128 Inception_ResNet_v2
# ./gpu_analysis.sh localhost 128 Inception_ResNet_v2

# echo Inception_v1
# ./gpu_eval_fb.sh localhost 128 Inception_v1
# ./gpu_analysis.sh localhost 128 Inception_v1

# echo Inception_v2
# ./gpu_eval_fb.sh localhost 128 Inception_v2
# ./gpu_analysis.sh localhost 128 Inception_v2

# echo Inception_v3
# ./gpu_eval_fb.sh localhost 64 Inception_v3
# ./gpu_analysis.sh localhost 64 Inception_v3

# echo Inception_v4
# ./gpu_eval_fb.sh localhost 128 Inception_v4
# ./gpu_analysis.sh localhost 128 Inception_v4

# echo MLPerf_Mobilenet_v1
# ./gpu_eval_fb.sh localhost 128 MLPerf_Mobilenet_v1
# ./gpu_analysis.sh localhost 128 MLPerf_Mobilenet_v1

# echo MLPerf_ResNet50_v1.5
# ./gpu_eval_fb.sh localhost 256 MLPerf_ResNet50_v1.5
# ./gpu_analysis.sh localhost 256 MLPerf_ResNet50_v1.5

# echo MLPerf_SSD_MobileNet_v1_300x300
# ./gpu_eval_fb.sh localhost 8 MLPerf_SSD_MobileNet_v1_300x300
# ./gpu_analysis.sh localhost 8 MLPerf_SSD_MobileNet_v1_300x300

# echo ResNet_v1_101
# ./gpu_eval_fb.sh localhost 256 ResNet_v1_101
# ./gpu_analysis.sh localhost 256 ResNet_v1_101

# echo ResNet_v2_101
# ./gpu_eval_fb.sh localhost 256 ResNet_v2_101
# ./gpu_analysis.sh localhost 256 ResNet_v2_101

# echo ResNet_v1_50
# ./gpu_eval_fb.sh localhost 256 ResNet_v1_50
# ./gpu_analysis.sh localhost 256 ResNet_v1_50

# echo ResNet_v2_50
# ./gpu_eval_fb.sh localhost 256 ResNet_v2_50
# ./gpu_analysis.sh localhost 256 ResNet_v2_50

# echo ResNet_v1_152
# ./gpu_eval_fb.sh localhost 256 ResNet_v1_152
# ./gpu_analysis.sh localhost 256 ResNet_v1_152

# echo ResNet_v2_50
# ./gpu_eval_fb.sh localhost 256 ResNet_v2_50
# ./gpu_analysis.sh localhost 256 ResNet_v2_50

# echo VGG16
# ./gpu_eval_fb.sh localhost 256 VGG16
# ./gpu_analysis.sh localhost 256 VGG16

# echo VGG19
# ./gpu_eval_fb.sh localhost 256 VGG19
# ./gpu_analysis.sh localhost 256 VGG19

# echo MobileNet_v1_0.25_128
# ./gpu_eval_fb.sh localhost 256 MobileNet_v1_0.25_128
# ./gpu_analysis.sh localhost 256 MobileNet_v1_0.25_128

# echo MobileNet_v1_0.25_160
# ./gpu_eval_fb.sh localhost 256 MobileNet_v1_0.25_160
# ./gpu_analysis.sh localhost 256 MobileNet_v1_0.25_160

# echo MobileNet_v1_0.25_192
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.25_192
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.25_192

# echo MobileNet_v1_0.25_224
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.25_224
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.25_224

# echo MobileNet_v1_0.5_128
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.5_128
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.5_128

# echo MobileNet_v1_0.5_160
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.5_160
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.5_160

# echo MobileNet_v1_0.5_192
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.5_192
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.5_192

# echo MobileNet_v1_0.5_224
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.5_224
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.5_224

# echo MobileNet_v1_0.75_128
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.75_128
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.75_128

# echo MobileNet_v1_0.75_160
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.75_160
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.75_160

# echo MobileNet_v1_0.75_192
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.75_192
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.75_192

# echo MobileNet_v1_0.75_224
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_0.75_224
# ./gpu_analysis.sh localhost 64 MobileNet_v1_0.75_224

# echo MobileNet_v1_1.0_128
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_1.0_128
# ./gpu_analysis.sh localhost 64 MobileNet_v1_1.0_128

# echo MobileNet_v1_1.0_160
# ./gpu_eval_fb.sh localhost 64 MobileNet_v1_1.0_160
# ./gpu_analysis.sh localhost 64 MobileNet_v1_1.0_160

# echo MobileNet_v1_1.0_192
# ./gpu_eval_fb.sh localhost 128 MobileNet_v1_1.0_192
# ./gpu_analysis.sh localhost 128 MobileNet_v1_1.0_192

# echo MobileNet_v1_1.0_224
# ./gpu_eval_fb.sh localhost 128 MobileNet_v1_1.0_224
# ./gpu_analysis.sh localhost 128 MobileNet_v1_1.0_224
