package benchmark

import (
	"runtime"
	"runtime/debug"
	"testing"
	"time"
	"unsafe"
)

// A global "black-hole" stops the compiler from eliminating work or keeping data on the stack.
var (
	sinkSlice []byte
	_         = unsafe.Sizeof(sinkSlice)
)

//go:noinline
func blackHole(b []byte) {
	sinkSlice = b // force the slice to escape
}

func BenchmarkWithGC(b *testing.B) {
	// Ensure GC is enabled with default settings
	debug.SetGCPercent(100)
	runtime.GC()

	run(b)
}

func BenchmarkWithoutGC(b *testing.B) {
	// Disable GC for benchmark
	originalGCPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	run(b)
}

func BenchmarkWithTunedGC(b *testing.B) {
	// Set a high GC threshold
	originalGCPercent := debug.SetGCPercent(500)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	run(b)
}

func allocateAndProcess(size int) int {
	data := make([]byte, size)

	// Do some work with the data
	sum := 0
	for i := 0; i < len(data); i++ {
		data[i] = byte(i % 256)
		sum += int(data[i])
	}

	blackHole(data)
	return sum
}

func run(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	// Measure latency
	var minLatency, maxLatency, totalLatency time.Duration

	for i := 0; i < b.N; i++ {
		start := time.Now()
		_ = allocateAndProcess(1024)
		latency := time.Since(start)

		if i == 0 || latency < minLatency {
			minLatency = latency
		}
		if latency > maxLatency {
			maxLatency = latency
		}

		totalLatency += latency
	}

	// Report latency distribution
	b.ReportMetric(float64(minLatency.Nanoseconds()), "min-ns")
	b.ReportMetric(float64(maxLatency.Nanoseconds()), "max-ns")
	b.ReportMetric(float64(totalLatency.Nanoseconds())/float64(b.N), "avg-ns")
}
