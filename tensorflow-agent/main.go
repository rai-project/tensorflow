package main

import (
	"fmt"
	"os"

	cmd "github.com/rai-project/dlframework/framework/cmd/server"
	"github.com/rai-project/tensorflow"
	_ "github.com/rai-project/tensorflow/predict"
	"github.com/rai-project/tracer"
)

func main() {

	rootCmd, err := cmd.NewRootCommand(tensorflow.FrameworkManifest)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	defer tracer.Close()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}
