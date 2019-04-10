#!/bin/bash

freeze_graph \
  --input_graph=./slim/inception_v1_2016_08_28/inception_v1.pb \
  --input_checkpoint=./slim/inception_v1_2016_08_28/inception_v1.ckpt \
  --input_binary=true --output_graph=./slim/inception_v1_2016_08_28/frozen_inception_v1.pb \
  --output_node_names=InceptionV1/Logits/Predictions/Reshape_1

freeze_graph \
  --input_graph=./slim/inception_v2_2016_08_28/inception_v2.pb \
  --input_checkpoint=./slim/inception_v2_2016_08_28/inception_v2.ckpt \
  --input_binary=true --output_graph=./slim/inception_v2_2016_08_28/frozen_inception_v2.pb \
  --output_node_names=InceptionV2/Predictions/Reshape_1

freeze_graph \
  --input_graph=./slim/inception_v3_2016_08_28/inception_v3.pb \
  --input_checkpoint=./slim/inception_v3_2016_08_28/inception_v3.ckpt \
  --input_binary=true --output_graph=./slim/inception_v3_2016_08_28/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Logits/Predictions/Reshape_1

freeze_graph \
  --input_graph=./slim/inception_v4_2016_09_09/inception_v4.pb \
  --input_checkpoint=./slim/inception_v4_2016_09_09/inception_v4.ckpt \
  --input_binary=true --output_graph=./slim/inception_v4_2016_09_09/frozen_inception_v4.pb \
  --output_node_names=InceptionV4/Logits/Predictions

freeze_graph \
  --input_graph=./slim/inception_resnet_v2_2016_08_30/inception_resnet_v2.pb \
  --input_checkpoint=./slim/inception_resnet_v2_2016_08_30/inception_resnet_v2.ckpt \
  --input_binary=true --output_graph=./slim/inception_resnet_v2_2016_08_30/frozen_inception_resnet_v2.pb \
  --output_node_names=InceptionResnetV2/Logits/Predictions

freeze_graph \
  --input_graph=./slim/resnet_v2_50_2017_04_14/resnet_v2_50.pb \
  --input_checkpoint=./slim/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt \
  --input_binary=true --output_graph=./slim/resnet_v2_50_2017_04_14/frozen_resnet_v2_50.pb \
  --output_node_names=resnet_v2_50/predictions/Reshape_1

freeze_graph \
  --input_graph=./slim/resnet_v2_101_2017_04_14/resnet_v2_101.pb \
  --input_checkpoint=./slim/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt \
  --input_binary=true --output_graph=./slim/resnet_v2_101_2017_04_14/frozen_resnet_v2_101.pb \
  --output_node_names=resnet_v2_101/predictions/Reshape_1

freeze_graph \
  --input_graph=./slim/resnet_v2_152_2017_04_14/resnet_v2_152.pb \
  --input_checkpoint=./slim/resnet_v2_152_2017_04_14/resnet_v2_152.ckpt \
  --input_binary=true --output_graph=./slim/resnet_v2_152_2017_04_14/frozen_resnet_v2_152.pb \
  --output_node_names=resnet_v2_152/predictions/Reshape_1

freeze_graph \
  --input_graph=./slim/vgg_16_2016_08_28/vgg_16.pb \
  --input_checkpoint=./slim/vgg_16_2016_08_28/vgg_16.ckpt \
  --input_binary=true --output_graph=./slim/vgg_16_2016_08_28/frozen_vgg_16.pb \
  --output_node_names=vgg_16/fc8/squeezed


