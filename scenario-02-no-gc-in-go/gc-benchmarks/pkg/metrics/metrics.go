package metrics

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"time"
)

// LatencyRecorder tracks operation latencies
type LatencyRecorder struct {
	Latencies []time.Duration
	StartTime time.Time
}

// NewLatencyRecorder creates a new latency recorder
func NewLatencyRecorder() *LatencyRecorder {
	return &LatencyRecorder{
		Latencies: make([]time.Duration, 0, 100_000),
		StartTime: time.Now(),
	}
}

// Record adds a latency measurement
func (lr *LatencyRecorder) Record(start time.Time) {
	lr.Latencies = append(lr.Latencies, time.Since(start))
}

// RecordFunc measures and records the execution time of a function
func (lr *LatencyRecorder) RecordFunc(f func()) {
	start := time.Now()
	f()
	lr.Record(start)
}

// PrintStats outputs latency statistics
func (lr *LatencyRecorder) PrintStats(label string) {
	if len(lr.Latencies) == 0 {
		fmt.Printf("%s: No latencies recorded\n", label)
		return
	}
	sort.Slice(lr.Latencies, func(i, j int) bool { return lr.Latencies[i] < lr.Latencies[j] })

	total := len(lr.Latencies)
	var sum time.Duration
	for _, lat := range lr.Latencies {
		sum += lat
	}

	mean := sum / time.Duration(total)
	p50 := percentile(lr.Latencies, 0.50)
	p90 := percentile(lr.Latencies, 0.90)
	p99 := percentile(lr.Latencies, 0.99)
	p999 := percentile(lr.Latencies, 0.999)
	maxLat := lr.Latencies[total-1]

	pureProcessing := sum.Seconds()
	throughput := float64(total) / pureProcessing

	fmt.Printf("\n=== %s ===\n", label)
	fmt.Printf("Operations:  %d\n", total)
	fmt.Printf("Duration:    %v\n", time.Since(lr.StartTime))
	fmt.Printf("Throughput:  %.2f ops/sec\n", throughput)
	fmt.Printf("Mean:        %v\n", mean)
	fmt.Printf("Median (p50): %v\n", p50)
	fmt.Printf("p90:         %v\n", p90)
	fmt.Printf("p99:         %v\n", p99)
	fmt.Printf("p99.9:       %v\n", p999)
	fmt.Printf("Max:         %v\n", maxLat)
}

// MemStats returns current memory statistics
func MemStats() runtime.MemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m
}

// PrintMemStats outputs memory usage information
func PrintMemStats(label string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Printf("\n=== Memory Stats: %s ===\n", label)
	fmt.Printf("Heap Alloc:   %d MB\n", m.HeapAlloc/1024/1024)
	fmt.Printf("Sys Memory:   %d MB\n", m.Sys/1024/1024)
	fmt.Printf("GC Cycles:    %d\n", m.NumGC)
	if m.NumGC > 0 {
		fmt.Printf("Last GC Pause: %v\n", time.Duration(m.PauseNs[(m.NumGC-1)%256]))
		fmt.Printf("Total GC Pause: %v\n", time.Duration(m.PauseTotalNs))
	}
}

// percentile returns the q‑percentile in [0,1] using linear interpolation.
func percentile(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	if q <= 0 {
		return sorted[0]
	}
	if q >= 1 {
		return sorted[len(sorted)-1]
	}
	pos := q * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	frac := pos - float64(lo)
	return sorted[lo] + time.Duration(frac*float64(sorted[hi]-sorted[lo]))
}
