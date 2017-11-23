package main

import (
	"errors"
	"time"

	opentracing "github.com/opentracing/opentracing-go"
	proto "github.com/rai-project/tensorflow"
	"golang.org/x/net/context"
)

type Trace struct {
	device string
	nodes  []*proto.NodeExecStats
}

func NewTrace(data *proto.StepStats) (*Trace, error) {
	if len(data.GetDevStats()) == 0 {
		return nil, errors.New("no device stats available")
	}
	fst := data.GetDevStats()[0]
	return &Trace{
		device: fst.GetDevice(),
		nodes:  fst.GetNodeStats(),
	}, nil
}

func (t *Trace) Publish(ctx context.Context, opts ...opentracing.StartSpanOption) error {
	for _, node := range t.nodes {

		startTime := time.Unix(0, node.GetAllStartMicros()*int64(time.Microsecond))
		endTime := time.Unix(0, (node.GetAllStartMicros()+node.GetAllEndRelMicros())*int64(time.Microsecond))

		tags := opentracing.Tags{
			"all_start_micros":    node.GetAllStartMicros(),
			"all_end_rel_micros":  node.GetAllEndRelMicros(),
			"op_start_rel_micros": node.GetOpStartRelMicros(),
			"op_end_rel_micros":   node.GetOpEndRelMicros(),
			"memory":              node.GetMemory(),
			"output":              node.GetOutput(),
			"timeline_label":      node.GetTimelineLabel(),
			"scheduled_micros":    node.GetScheduledMicros(),
			"referenced_tensor":   node.GetAllStartMicros(),
			"memory_stats":        node.GetMemoryStats(),
			"thread_id":           node.GetThreadId(),
			"start_time":          startTime,
			"end_time":            endTime,
		}

		s, _ := opentracing.StartSpanFromContext(
			ctx,
			node.GetNodeName(),
			opentracing.StartTime(startTime),
			tags,
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
