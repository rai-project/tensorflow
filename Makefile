all: generate

fmt:
	go fmt ./...

install-deps:
	go get github.com/jteeuwen/go-bindata/...
	go get github.com/elazarl/go-bindata-assetfs/...
	go get github.com/golang/dep
	dep ensure -v


generate: clean generate-models

generate-proto:
	protoc --plugin=protoc-gen-go=${GOPATH}/bin/protoc-gen-go -Iproto \
		--gogofaster_out=Mgoogle/protobuf/any.proto=github.com/gogo/protobuf/types,plugins=grpc:. \
		proto/allocation_description.proto proto/attr_value.proto proto/cluster.proto proto/config.proto \
		proto/control_flow.proto proto/cost_graph.proto proto/debug.proto proto/device_attributes.proto \
		proto/device_properties.proto proto/error_codes.proto proto/function.proto proto/graph.proto \
		proto/graph_transfer_info.proto proto/kernel_def.proto proto/log_memory.proto proto/master.proto \
		proto/master_service.proto proto/meta_graph.proto proto/named_tensor.proto proto/node_def.proto \
		proto/op_def.proto proto/op_gen_overrides.proto proto/queue_runner.proto proto/reader_base.proto \
		proto/remote_fused_graph_execute_info.proto proto/resource_handle.proto proto/rewriter_config.proto \
		proto/saved_model.proto proto/saver.proto proto/step_stats.proto proto/summary.proto proto/tensor_bundle.proto \
		proto/tensor_description.proto proto/tensorflow_server.proto proto/tensor.proto proto/tensor_shape.proto \
		proto/tensor_slice.proto proto/types.proto proto/variable.proto proto/versions.proto proto/worker.proto \
		proto/worker_service.proto

generate-models:
	go-bindata -nomemcopy -prefix builtin_models/ -pkg tensorflow -o builtin_models_static.go -ignore=.DS_Store  -ignore=README.md builtin_models/...

clean-models:
	rm -fr builtin_models_static.go

clean-proto:
	rm -fr *pb.go

clean: clean-models

travis: install-deps glide-install generate
	echo "building..."
	go build
