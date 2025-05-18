package benchmark

import (
	"runtime"
	"runtime/debug"
	"sync"
	"testing"
	"time"
)

func BenchmarkSyncPoolWithGC(b *testing.B) {
	// Ensure GC is enabled with default settings
	debug.SetGCPercent(100)
	runtime.GC()

	runSyncPool(b)
}

func BenchmarkSyncPoolWithoutGC(b *testing.B) {
	// Disable GC for benchmark
	originalGCPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	runSyncPool(b)
}

func BenchmarkSyncPoolWithTunedGC(b *testing.B) {
	// Set a high GC threshold
	originalGCPercent := debug.SetGCPercent(500)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	runSyncPool(b)
}

func newSyncPool(size int) *sync.Pool {
	return &sync.Pool{
		New: func() any {
			return make([]byte, size)
		},
	}
}

func processSyncPool(pool *sync.Pool) int {
	// Get data from the pool
	data := pool.Get().([]byte)
	defer pool.Put(data)

	// Do some work with the data
	sum := 0
	for i := 0; i < len(data); i++ {
		data[i] = byte(i % 256)
		sum += int(data[i])
	}

	return sum
}

func runSyncPool(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	pool := newSyncPool(1024)

	// Measure latency
	var minLatency, maxLatency, totalLatency time.Duration

	for i := 0; i < b.N; i++ {
		start := time.Now()
		_ = processSyncPool(pool)
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
