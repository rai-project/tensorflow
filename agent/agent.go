package agent

import (
	rgrpc "github.com/rai-project/grpc"
	"github.com/rai-project/uuid"
	context "golang.org/x/net/context"
	"google.golang.org/grpc"

	dl "github.com/rai-project/dlframework"
	common "github.com/rai-project/dlframework/framework/agent"
	tf "github.com/rai-project/tensorflow"
	predict "github.com/rai-project/tensorflow/predict"
)

type registryServer struct {
	common.Registry
}

type predictorServer struct {
	common.Predictor
}

func (p *predictorServer) Predict(ctx context.Context, req *dl.PredictRequest) (*dl.PredictResponse, error) {
	_, model, err := p.FindFrameworkModel(ctx, req)
	if err != nil {
		return nil, err
	}

	predictor, err := predict.New(*model)
	if err != nil {
		return nil, err
	}
	defer predictor.Close()
	if err := predictor.Download(); err != nil {
		return nil, err
	}

	reader, err := p.InputReaderCloser(ctx, req)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	data, err := predictor.Preprocess(reader)
	if err != nil {
		return nil, err
	}

	probs, err := predictor.Predict(data)
	if err != nil {
		return nil, err
	}

	probs.Sort()

	if req.GetLimit() != 0 {
		trunc := probs.Take(int(req.GetLimit()))
		probs = &trunc
	}

	return &dl.PredictResponse{
		Id:       uuid.NewV4(),
		Features: *probs,
		Error:    nil,
	}, nil
}

func RegisterRegistryServer() *grpc.Server {
	var grpcServer *grpc.Server
	grpcServer = rgrpc.NewServer(dl.RegistryServiceDescription)
	svr := &registryServer{
		Registry: common.Registry{
			Base: common.Base{
				Framework: tf.FrameworkManifest,
			},
		},
	}
	svr.PublishInRegistery()
	dl.RegisterRegistryServer(grpcServer, svr)
	return grpcServer
}

func RegisterPredictorServer() *grpc.Server {
	var grpcServer *grpc.Server
	grpcServer = rgrpc.NewServer(dl.PredictorServiceDescription)
	svr := &predictorServer{
		Predictor: common.Predictor{
			Base: common.Base{
				Framework: tf.FrameworkManifest,
			},
		},
	}
	svr.PublishInRegistery()
	dl.RegisterPredictorServer(grpcServer, svr)
	return grpcServer
}
