//go:generate go get -v github.com/jteeuwen/go-bindata/...
//go:generate go get -v github.com/elazarl/go-bindata-assetfs/...
//go:generate go-bindata -nomemcopy -pkg main -o static.go -ignore=.DS_Store _fixtures/...

package graph
