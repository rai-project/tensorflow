//go:generate go get -v github.com/jteeuwen/go-bindata/...
//go:generate go get -v github.com/elazarl/go-bindata-assetfs/...
//go:generate go-bindata -nomemcopy -prefix builtin_models/ -pkg tensorflow -o builtin_models_static.go -ignore=.DS_Store  -ignore=README.md builtin_models/...
//go:generate go-bindata -nomemcopy -prefix graph/ -pkg main -o graph/static.go -ignore=.DS_Store graph/_fixtures/...

package tensorflow
