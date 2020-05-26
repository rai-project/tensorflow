#! /bin/bash

svn export --force https://github.com/rai-project/mlmodelscope/trunk/carml_config.yml.example ~/.carml_config.yml

docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest

docker run --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --privileged=true \
   --network host -v ~/.carml_config.yml:/root/.carml_config.yml $1 predict urls
