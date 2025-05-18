package benchmark

import (
	"runtime"
	"runtime/debug"
	"testing"
	"time"
)

func BenchmarkChanPoolWithGC(b *testing.B) {
	// Ensure GC is enabled with default settings
	debug.SetGCPercent(100)
	runtime.GC()

	runChanPool(b)
}

func BenchmarkChanPoolWithoutGC(b *testing.B) {
	// Disable GC for benchmark
	originalGCPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	runChanPool(b)
}

func BenchmarkChanPoolWithTunedGC(b *testing.B) {
	// Set a high GC threshold
	originalGCPercent := debug.SetGCPercent(500)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	runChanPool(b)
}

type chanPool chan []byte

func newChanPool(size, capacity int) chanPool {
	ch := make(chan []byte, capacity)
	for i := 0; i < capacity; i++ {
		ch <- make([]byte, size)
	}
	return ch
}

func processChanPool(pool chanPool) int {
	data := <-pool
	defer func() { pool <- data }()

	sum := 0
	for i := 0; i < len(data); i++ {
		data[i] = byte(i % 256)
		sum += int(data[i])
	}

	return sum
}

func runChanPool(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	pool := newChanPool(1024, 5)

	// Measure latency
	var minLatency, maxLatency, totalLatency time.Duration

	for i := 0; i < b.N; i++ {
		start := time.Now()
		_ = processChanPool(pool)
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
