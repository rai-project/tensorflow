package predictor

func zeros(height, width, channels int) [][][]float32 {
	rows := make([][][]float32, height)
	for ii := range rows {
		columns := make([][]float32, width)
		for jj := range columns {
			columns[jj] = make([]float32, channels)
		}
		rows[ii] = columns
	}
	return rows
}

// func  createTensor(ctx context.Context, data [][]float32) (*tf.Tensor, error) {
// 	span, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "create_tensor")
// 	defer span.Finish()

// 	imageDims, err := p.GetImageDimensions()
// 	if err != nil {
// 		return nil, err
// 	}

// 	channels, height, width := int64(imageDims[0]), int64(imageDims[1]), int64(imageDims[2])
// 	batchSize := int64(p.BatchSize())
// 	if batchSize == 0 {
// 		batchSize = 1
// 	}

// 	shapeLen := width * height * channels
// 	dataLen := int64(len(data))
// 	if batchSize > dataLen {
// 		padding := make([]float32, (batchSize-dataLen)*shapeLen)
// 		data = append(data, padding)
// 	}

// 	return NewTensor(ctx, data, []int64{batchSize, height, width, channels})
// }
