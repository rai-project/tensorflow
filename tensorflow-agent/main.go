package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/agent"
	cmd "github.com/rai-project/dlframework/framework/cmd/server"
	"github.com/rai-project/tensorflow"
	_ "github.com/rai-project/tensorflow/predictor"
	"github.com/rai-project/tracer"
	"github.com/spf13/cobra"
)

var (
	modelName    string
	modelVersion string
	hostName, _  = os.HostName()
	framework    = tensorflow.FrameworkManifest
)

func graphConvert(c *cobra.Command, args []string) error {
	ctx := context.Background()

	predictorsFramework, err := agent.GetPredictors(framework)
	if err != nil {
		return errors.Wrapf(err,
			"⚠️ failed to get predictor for %s. make sure you have "+
				"imported the framework's predictor package",
			framework.MustCanonicalName(),
		)
	}

	// really does not matter the type of predictor
	var predictorFramework tensorflow.ImageClassificationPredictor
	for _, p := range predictorsFramework {
		if s, ok := p.(*tensorflow.ImageClassificationPredictor); ok {
			predictorFramework = s
		}
	}
	if predictorFramework == nil {
		return errors.Wrapf(err,
			"⚠️ failed to get predictor for %s. make sure you have "+
				"imported the framework's predictor package..",
			framework.MustCanonicalName(),
		)
	}

	model, err := framework.FindModel(modelName + ":" + modelVersion)
	if err != nil {
		return err
	}

	err := predictorFramework.Download(ctx, model)
	if err != nil {
		return errors.Wrapf(err, "failed to download %s model", model.MustCanonicalName())
	}

	g, err := graph.New()
	if err != nil {
		panic(err)
	}

	bts, err := g.MarshalJSON()
	if err != nil {
		panic(err)
	}

	baseDir := filepath.Join("experiments", hostName, framework.Name, framework.Version, model.Name, model.Version)
	if !com.IsDir(baseDir) {
		os.MkdirAll(baseDir, os.ModePerm)
	}

	ioutils.WriteFile(fielpath.Join(baseDir, "model_info.json"), bts, 0644)

	return err
}

var graphCmd = &cobra.Command{
	Use:   "model_graph",
	Short: "Converts the frozen graph into a model graph summary",
	RunE: func(c *cobra.Command, args []string) error {
		if modelName == "all" {
			for _, model := range framework.Models() {
				modelName = model.Name
				modelVersion = model.Version
				graphConvert(c, args)
			}
			return nil
		}
		return graphConvert(c, args)
	},
}

func init() {
	graphCmd.PersistentFlags().StringVar(&modelName, "model_name", "MobileNet_v1_1.0_224", "the name of the model to use for conversion")
	graphCmd.PersistentFlags().StringVar(&modelVersion, "model_version", "1.0", "the version of the model to use for conversion")
}

func main() {
	rootCmd, err := cmd.NewRootCommand(tensorflow.Register, framework)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	rootCmd.Add(graphCmd)

	defer tracer.Close()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}
