

mkdir -p bvlc_alexnet_1.0
cd bvlc_alexnet_1.0

echo "Converting BVLC-Alexnet"
python -m mmdnn.conversion._script.convertToIR -f caffe -d BVLC-Alexnet -n ~/data/carml/dlframework/caffe_1.0/bvlc_alexnet_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/bvlc_alexnet_1.0/bvlc_alexnet.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath BVLC-Alexnet.pb  --IRWeightPath BVLC-Alexnet.npy --dstModelPath tf_BVLC-Alexnet.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_BVLC-Alexnet.py -w BVLC-Alexnet.npy  --dump ./BVLC-Alexnet.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting BVLC-Alexnet"

cd ..


mkdir -p bvlc_reference_caffenet_1.0
cd bvlc_reference_caffenet_1.0

echo "Converting BVLC-Reference-CaffeNet"
python -m mmdnn.conversion._script.convertToIR -f caffe -d BVLC-Reference-CaffeNet -n ~/data/carml/dlframework/caffe_1.0/bvlc_reference_caffenet_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/bvlc_reference_caffenet_1.0/bvlc_reference_caffenet.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath BVLC-Reference-CaffeNet.pb  --IRWeightPath BVLC-Reference-CaffeNet.npy --dstModelPath tf_BVLC-Reference-CaffeNet.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_BVLC-Reference-CaffeNet.py -w BVLC-Reference-CaffeNet.npy  --dump ./BVLC-Reference-CaffeNet.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting BVLC-Reference-CaffeNet"

cd ..


mkdir -p bvlc_reference_rcnn_ilsvrc13_1.0
cd bvlc_reference_rcnn_ilsvrc13_1.0

echo "Converting BVLC-Reference-RCNN_ILSVRC13"
python -m mmdnn.conversion._script.convertToIR -f caffe -d BVLC-Reference-RCNN_ILSVRC13 -n ~/data/carml/dlframework/caffe_1.0/bvlc_reference_rcnn_ilsvrc13_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/bvlc_reference_rcnn_ilsvrc13_1.0/bvlc_reference_rcnn_ilsvrc13.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath BVLC-Reference-RCNN_ILSVRC13.pb  --IRWeightPath BVLC-Reference-RCNN_ILSVRC13.npy --dstModelPath tf_BVLC-Reference-RCNN_ILSVRC13.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_BVLC-Reference-RCNN_ILSVRC13.py -w BVLC-Reference-RCNN_ILSVRC13.npy  --dump ./BVLC-Reference-RCNN_ILSVRC13.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting BVLC-Reference-RCNN_ILSVRC13"

cd ..


mkdir -p dpn68_1.0
cd dpn68_1.0

echo "Converting DPN68"
python -m mmdnn.conversion._script.convertToIR -f caffe -d DPN68 -n ~/data/carml/dlframework/caffe_1.0/dpn68_1.0/deploy_dpn68_extra_no_ceil_mode.prototxt -w ~/data/carml/dlframework/caffe_1.0/dpn68_1.0/dpn68_extra.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath DPN68.pb  --IRWeightPath DPN68.npy --dstModelPath tf_DPN68.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_DPN68.py -w DPN68.npy  --dump ./DPN68.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting DPN68"

cd ..


mkdir -p dpn92_1.0
cd dpn92_1.0

echo "Converting DPN92"
python -m mmdnn.conversion._script.convertToIR -f caffe -d DPN92 -n ~/data/carml/dlframework/caffe_1.0/dpn92_1.0/deploy_dpn92_no_ceil_mode.prototxt -w ~/data/carml/dlframework/caffe_1.0/dpn92_1.0/dpn92.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath DPN92.pb  --IRWeightPath DPN92.npy --dstModelPath tf_DPN92.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_DPN92.py -w DPN92.npy  --dump ./DPN92.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting DPN92"

cd ..


mkdir -p inception_3.0
cd inception_3.0

echo "Converting Inception_3_0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d Inception_3_0 -n ~/data/carml/dlframework/caffe_1.0/inception_3.0/deploy_inception_v3.prototxt -w ~/data/carml/dlframework/caffe_1.0/inception_3.0/inception_v3.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath Inception_3_0.pb  --IRWeightPath Inception_3_0.npy --dstModelPath tf_Inception_3_0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_Inception_3_0.py -w Inception_3_0.npy  --dump ./Inception_3_0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting Inception_3_0"

cd ..


mkdir -p inception_4.0
cd inception_4.0

echo "Converting Inception_4_0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d Inception_4_0 -n ~/data/carml/dlframework/caffe_1.0/inception_4.0/deploy_inception_v4.prototxt -w ~/data/carml/dlframework/caffe_1.0/inception_4.0/inception_v4.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath Inception_4_0.pb  --IRWeightPath Inception_4_0.npy --dstModelPath tf_Inception_4_0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_Inception_4_0.py -w Inception_4_0.npy  --dump ./Inception_4_0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting Inception_4_0"

cd ..


mkdir -p inceptionbn_21k_2.0
cd inceptionbn_21k_2.0

echo "Converting InceptionBN_21K"
python -m mmdnn.conversion._script.convertToIR -f caffe -d InceptionBN_21K -n ~/data/carml/dlframework/caffe_1.0/inceptionbn_21k_2.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/inceptionbn_21k_2.0/inception21k.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath InceptionBN_21K.pb  --IRWeightPath InceptionBN_21K.npy --dstModelPath tf_InceptionBN_21K.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_InceptionBN_21K.py -w InceptionBN_21K.npy  --dump ./InceptionBN_21K.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting InceptionBN_21K"

cd ..


mkdir -p inception_resnet_2.0
cd inception_resnet_2.0

echo "Converting InceptionResnet"
python -m mmdnn.conversion._script.convertToIR -f caffe -d InceptionResnet -n ~/data/carml/dlframework/caffe_1.0/inception_resnet_2.0/deploy_inception_resnet_v2.prototxt -w ~/data/carml/dlframework/caffe_1.0/inception_resnet_2.0/inception_resnet_v2.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath InceptionResnet.pb  --IRWeightPath InceptionResnet.npy --dstModelPath tf_InceptionResnet.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_InceptionResnet.py -w InceptionResnet.npy  --dump ./InceptionResnet.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting InceptionResnet"

cd ..


mkdir -p network_in_network_1.0
cd network_in_network_1.0

echo "Converting NIN"
python -m mmdnn.conversion._script.convertToIR -f caffe -d NIN -n ~/data/carml/dlframework/caffe_1.0/network_in_network_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/network_in_network_1.0/nin_imagenet.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath NIN.pb  --IRWeightPath NIN.npy --dstModelPath tf_NIN.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_NIN.py -w NIN.npy  --dump ./NIN.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting NIN"

cd ..


mkdir -p resnet101_1.0
cd resnet101_1.0

echo "Converting Resnet101_1_0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d Resnet101_1_0 -n ~/data/carml/dlframework/caffe_1.0/resnet101_1.0/resnet_101_deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/resnet101_1.0/resnet_101_model.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath Resnet101_1_0.pb  --IRWeightPath Resnet101_1_0.npy --dstModelPath tf_Resnet101_1_0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_Resnet101_1_0.py -w Resnet101_1_0.npy  --dump ./Resnet101_1_0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting Resnet101_1_0"

cd ..


mkdir -p resnet101_2.0
cd resnet101_2.0

echo "Converting Resnet101_2_0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d Resnet101_2_0 -n ~/data/carml/dlframework/caffe_1.0/resnet101_2.0/deploy_resnet101_v2.prototxt -w ~/data/carml/dlframework/caffe_1.0/resnet101_2.0/resnet101_v2.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath Resnet101_2_0.pb  --IRWeightPath Resnet101_2_0.npy --dstModelPath tf_Resnet101_2_0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_Resnet101_2_0.py -w Resnet101_2_0.npy  --dump ./Resnet101_2_0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting Resnet101_2_0"

cd ..


mkdir -p resnet152_1.0
cd resnet152_1.0

echo "Converting Resnet152_1_0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d Resnet152_1_0 -n ~/data/carml/dlframework/caffe_1.0/resnet152_1.0/resnet_152_deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/resnet152_1.0/resnet_152_model.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath Resnet152_1_0.pb  --IRWeightPath Resnet152_1_0.npy --dstModelPath tf_Resnet152_1_0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_Resnet152_1_0.py -w Resnet152_1_0.npy  --dump ./Resnet152_1_0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting Resnet152_1_0"

cd ..


mkdir -p resnet152_2.0
cd resnet152_2.0

echo "Converting Resnet152_2_0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d Resnet152_2_0 -n ~/data/carml/dlframework/caffe_1.0/resnet152_2.0/deploy_resnet152_v2.prototxt -w ~/data/carml/dlframework/caffe_1.0/resnet152_2.0/resnet152_v2.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath Resnet152_2_0.pb  --IRWeightPath Resnet152_2_0.npy --dstModelPath tf_Resnet152_2_0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_Resnet152_2_0.py -w Resnet152_2_0.npy  --dump ./Resnet152_2_0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting Resnet152_2_0"

cd ..


mkdir -p resnet18_1.0
cd resnet18_1.0

echo "Converting resnet18_1.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d resnet18_1.0 -n ~/data/carml/dlframework/caffe_1.0/resnet18_1.0/deploy_resnet18_priv_no_ceil_mode.prototxt -w ~/data/carml/dlframework/caffe_1.0/resnet18_1.0/resnet18_priv.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath resnet18_1.0.pb  --IRWeightPath resnet18_1.0.npy --dstModelPath tf_resnet18_1.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_resnet18_1.0.py -w resnet18_1.0.npy  --dump ./resnet18_1.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting resnet18_1.0"

cd ..


mkdir -p resnet269_2.0
cd resnet269_2.0

echo "Converting resnet269_2.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d resnet269_2.0 -n ~/data/carml/dlframework/caffe_1.0/resnet269_2.0/deploy_resnet269_v2.prototxt -w ~/data/carml/dlframework/caffe_1.0/resnet269_2.0/resnet269_v2.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath resnet269_2.0.pb  --IRWeightPath resnet269_2.0.npy --dstModelPath tf_resnet269_2.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_resnet269_2.0.py -w resnet269_2.0.npy  --dump ./resnet269_2.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting resnet269_2.0"

cd ..


mkdir -p resnext50_32x4d_1.0
cd resnext50_32x4d_1.0

echo "Converting resnext50_32x4d_1.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d resnext50_32x4d_1.0 -n ~/data/carml/dlframework/caffe_1.0/resnext50_32x4d_1.0/deploy_resnext50_32x4d_no_ceil_mode.prototxt -w ~/data/carml/dlframework/caffe_1.0/resnext50_32x4d_1.0/resnext50_32x4d.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath resnext50_32x4d_1.0.pb  --IRWeightPath resnext50_32x4d_1.0.npy --dstModelPath tf_resnext50_32x4d_1.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_resnext50_32x4d_1.0.py -w resnext50_32x4d_1.0.npy  --dump ./resnext50_32x4d_1.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting resnext50_32x4d_1.0"

cd ..


mkdir -p squeezenet_1.0
cd squeezenet_1.0

echo "Converting squeezenet_1.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d squeezenet_1.0 -n ~/data/carml/dlframework/caffe_1.0/squeezenet_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/squeezenet_1.0/squeezenet_v1.0.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath squeezenet_1.0.pb  --IRWeightPath squeezenet_1.0.npy --dstModelPath tf_squeezenet_1.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_squeezenet_1.0.py -w squeezenet_1.0.npy  --dump ./squeezenet_1.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting squeezenet_1.0"

cd ..


mkdir -p squeezenet_1.1
cd squeezenet_1.1

echo "Converting squeezenet_1.1"
python -m mmdnn.conversion._script.convertToIR -f caffe -d squeezenet_1.1 -n ~/data/carml/dlframework/caffe_1.0/squeezenet_1.1/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/squeezenet_1.1/squeezenet_v1.1.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath squeezenet_1.1.pb  --IRWeightPath squeezenet_1.1.npy --dstModelPath tf_squeezenet_1.1.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_squeezenet_1.1.py -w squeezenet_1.1.npy  --dump ./squeezenet_1.1.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting squeezenet_1.1"

cd ..


mkdir -p vgg16_1.0
cd vgg16_1.0

echo "Converting vgg16_1.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d vgg16_1.0 -n ~/data/carml/dlframework/caffe_1.0/vgg16_1.0/vgg_ilsvrc_16_layers_deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/vgg16_1.0/vgg_ilsvrc_16_layers.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath vgg16_1.0.pb  --IRWeightPath vgg16_1.0.npy --dstModelPath tf_vgg16_1.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_vgg16_1.0.py -w vgg16_1.0.npy  --dump ./vgg16_1.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting vgg16_1.0"

cd ..


mkdir -p vgg16_sod_1.0
cd vgg16_sod_1.0

echo "Converting vgg16_sod_1.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d vgg16_sod_1.0 -n ~/data/carml/dlframework/caffe_1.0/vgg16_sod_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/vgg16_sod_1.0/vgg16_sod_finetune.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath vgg16_sod_1.0.pb  --IRWeightPath vgg16_sod_1.0.npy --dstModelPath tf_vgg16_sod_1.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_vgg16_sod_1.0.py -w vgg16_sod_1.0.npy  --dump ./vgg16_sod_1.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting vgg16_sod_1.0"

cd ..


mkdir -p vgg16_sos_1.0
cd vgg16_sos_1.0

echo "Converting vgg16_sos_1.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d vgg16_sos_1.0 -n ~/data/carml/dlframework/caffe_1.0/vgg16_sos_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/vgg16_sos_1.0/vgg16_salobjsub.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath vgg16_sos_1.0.pb  --IRWeightPath vgg16_sos_1.0.npy --dstModelPath tf_vgg16_sos_1.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_vgg16_sos_1.0.py -w vgg16_sos_1.0.npy  --dump ./vgg16_sos_1.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting vgg16_sos_1.0"

cd ..


mkdir -p vgg19_1.0
cd vgg19_1.0

echo "Converting vgg19_1.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d vgg19_1.0 -n ~/data/carml/dlframework/caffe_1.0/vgg19_1.0/vgg_ilsvrc_19_layers_deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/vgg19_1.0/vgg_ilsvrc_19_layers.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath vgg19_1.0.pb  --IRWeightPath vgg19_1.0.npy --dstModelPath tf_vgg19_1.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_vgg19_1.0.py -w vgg19_1.0.npy  --dump ./vgg19_1.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting vgg19_1.0"

cd ..


mkdir -p wrn50_2.0
cd wrn50_2.0

echo "Converting wrn50_2.0"
python -m mmdnn.conversion._script.convertToIR -f caffe -d wrn50_2.0 -n ~/data/carml/dlframework/caffe_1.0/wrn50_2.0/deploy_wrn50_2_no_ceil_mode.prototxt -w ~/data/carml/dlframework/caffe_1.0/wrn50_2.0/wrn50_2.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath wrn50_2.0.pb  --IRWeightPath wrn50_2.0.npy --dstModelPath tf_wrn50_2.0.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_wrn50_2.0.py -w wrn50_2.0.npy  --dump ./wrn50_2.0.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting wrn50_2.0"

cd ..


mkdir -p bvlc_googlenet_1.0
cd bvlc_googlenet_1.0

echo "Converting BVLC-GoogLeNet"
python -m mmdnn.conversion._script.convertToIR -f caffe -d BVLC-GoogLeNet -n ~/data/carml/dlframework/caffe_1.0/bvlc_googlenet_1.0/deploy.prototxt -w ~/data/carml/dlframework/caffe_1.0/bvlc_googlenet_1.0/bvlc_googlenet.caffemodel

python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath BVLC-GoogLeNet.pb  --IRWeightPath BVLC-GoogLeNet.npy --dstModelPath tf_BVLC-GoogLeNet.py

python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_BVLC-GoogLeNet.py -w BVLC-GoogLeNet.npy  --dump ./BVLC-GoogLeNet.ckpt

python ../../medium-tffreeze-1.py  --model_dir=. --output_node_names prob

echo "Done Converting BVLC-GoogLeNet"

cd ..
 
 


