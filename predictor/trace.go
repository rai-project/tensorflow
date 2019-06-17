package predictor

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	opentracing "github.com/opentracing/opentracing-go"
	proto "github.com/rai-project/tensorflow"
	"github.com/rai-project/tracer"
)

type traceNode struct {
	device string
	node   *proto.NodeExecStats
}

type Trace struct {
	nodes []traceNode
}

func NewTrace(data *proto.StepStats) (*Trace, error) {
	if len(data.GetDevStats()) == 0 {
		return nil, errors.New("no device stats available")
	}
	nodes := []traceNode{}
	for _, dev := range data.GetDevStats() {
		for _, nd := range dev.GetNodeStats() {
			nodes = append(nodes, traceNode{
				device: dev.GetDevice(),
				node:   nd,
			})
		}
	}
	return &Trace{
		nodes: nodes,
	}, nil
}

// Notes about start and end time from the NodeExecStats proto:
// For GPU, there is no difference between op_end_rel_micros and
// all_end_rel_micros. All are kernel times.
// For CPU, op_end_rel is the kernel time, while all_end_rel_micros includes
// some post-processing. Besides, currently, there is no way to measure
// the execution time of async ops accurately.
func (t *Trace) Publish(ctx context.Context, opts ...opentracing.StartSpanOption) error {
	for layerSequenceIndex, tr := range t.nodes {
		device := tr.device
		node := tr.node
		startTime := time.Unix(0, node.GetAllStartMicros()*int64(time.Microsecond))
		endTime := time.Unix(0, (node.GetAllStartMicros()+node.GetAllEndRelMicros())*int64(time.Microsecond))
		if endTime.Sub(startTime) <= 10*time.Microsecond {
			continue
		}
		tags := opentracing.Tags{
			"trace_source":   "framework",
			"framework_name": "tensorflow",
			"device":         device,
			// "all_start_micros":    node.GetAllStartMicros(),
			// "all_end_rel_micros":  node.GetAllEndRelMicros(),
			// "op_start_rel_micros": node.GetOpStartRelMicros(),
			// "op_end_rel_micros":   node.GetOpEndRelMicros(),
			"timeline_label":   node.GetTimelineLabel(),
			"scheduled_micros": node.GetScheduledMicros(),
			"thread_id":        node.GetThreadId(),
			// "start_time":          startTime,
			// "end_time":            endTime,
			"layer_sequence_index": layerSequenceIndex,
		}
		if len(node.GetOutput()) != 0 {
			shapes := make([][]int64, len(node.GetOutput()))
			for jj, o := range node.GetOutput() {
				ss := o.GetTensorDescription().GetShape().GetDim()
				if len(ss) == 0 {
					shapes[jj] = []int64{}
					continue
				}
				shape := make([]int64, len(ss))
				for ii, s := range ss {
					shape[ii] = s.GetSize_()
				}
				shapes[jj] = shape
			}
			tags["shape"] = shapes
		}
		memStatsTags := opentracing.Tags{}
		if node.GetMemoryStats() != nil {
			stats := node.GetMemoryStats()
			memStatsTags = opentracing.Tags{
				"temp_memory_size":                   stats.GetTempMemorySize(),
				"persistent_memory_size":             stats.GetPersistentMemorySize(),
				"device_temp_memory_size":            stats.GetDeviceTempMemorySize(),
				"device_persistent_memory_size":      stats.GetDevicePersistentMemorySize(),
				"host_persistent_tensor_alloc_ids":   stats.GetPersistentTensorAllocIds(),
				"device_persistent_tensor_alloc_ids": stats.GetDevicePersistentTensorAllocIds(),
			}
		}

		memAllocTags := opentracing.Tags{}
		if node.GetMemory() != nil {
			stats := node.GetMemory()
			bts, err := json.Marshal(stats)
			if err != nil {
				break
			}
			memAllocTags = opentracing.Tags{
				"memory": string(bts),
			}
		}

		tensorTags := opentracing.Tags{}
		if node.GetReferencedTensor() != nil {
			stats := node.GetReferencedTensor()
			bts, err := json.Marshal(stats)
			if err != nil {
				break
			}
			memAllocTags = opentracing.Tags{
				"tensor": string(bts),
			}
		}

		outputTags := opentracing.Tags{}
		if node.GetOutput() != nil {
			stats := node.GetOutput()
			bts, err := json.Marshal(stats)
			if err != nil {
				break
			}
			memAllocTags = opentracing.Tags{
				"output": string(bts),
			}
		}

		s, _ := tracer.StartSpanFromContext(
			ctx,
			tracer.FRAMEWORK_TRACE,
			node.GetNodeName(),
			opentracing.StartTime(startTime),
			tags,
			memStatsTags,
			memAllocTags,
			tensorTags,
			outputTags,
		)
		if s == nil {
			log.WithField("node_name", node.GetNodeName()).
				WithField("tags", tags).
				Error("failed to create span from context")
			continue
		}
		s.FinishWithOptions(opentracing.FinishOptions{
			FinishTime: endTime,
		})
	}

	return nil
}
