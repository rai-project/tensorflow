package tensorflow

import (
	"os"

	assetfs "github.com/elazarl/go-bindata-assetfs"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/frameworks/common"
)

var FrameworkManifest = dlframework.FrameworkManifest{
	Name:    "Tensorflow",
	Version: "1.2",
	Container: map[string]*dlframework.ContainerHardware{
		"amd64": {
			Cpu: "raiproject/carml-tensorflow:amd64-cpu",
			Gpu: "raiproject/carml-tensorflow:amd64-gpu",
		},
		"ppc64le": {
			Cpu: "raiproject/carml-tensorflow:ppc64le-gpu",
			Gpu: "raiproject/carml-tensorflow:ppc64le-gpu",
		},
	},
}

func assetFS() *assetfs.AssetFS {
	assetInfo := func(path string) (os.FileInfo, error) {
		return os.Stat(path)
	}
	for k := range _bintree.Children {
		return &assetfs.AssetFS{Asset: Asset, AssetDir: AssetDir, AssetInfo: assetInfo, Prefix: k}
	}
	panic("unreachable")
}

func init() {
	common.Register(FrameworkManifest, assetFS())
}
