buildlocalppc:
	docker build . -f Dockerfile.ppc64le -t carml/tensorflow:base-ppc64le-gpu
buildlocalamd64gpu:
	docker build . -f Dockerfile.amd64_gpu -t carml/tensorflow:base-amd64-gpu
buildlocalamd64cpu:
	docker build . -f Dockerfile.amd64_cpu -t carml/tensorflow:base-amd64-cpu

buildlocalamd64: buildlocalamd64cpu buildlocalamd64gpu
