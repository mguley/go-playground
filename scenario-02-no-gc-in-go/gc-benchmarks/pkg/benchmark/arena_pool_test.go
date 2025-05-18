package benchmark

import (
	"runtime"
	"runtime/debug"
	"testing"
	"time"
)

func BenchmarkArenaPoolWithGC(b *testing.B) {
	// Ensure GC is enabled with default settings
	debug.SetGCPercent(100)
	runtime.GC()

	runArenaPool(b)
}

func BenchmarkArenaPoolWithoutGC(b *testing.B) {
	// Disable GC for benchmark
	originalGCPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	runArenaPool(b)
}

func BenchmarkArenaPoolWithTunedGC(b *testing.B) {
	// Set a high GC threshold
	originalGCPercent := debug.SetGCPercent(500)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	runArenaPool(b)
}

type slab struct {
	mem []byte // one big backing array
	pos int
}

func newSlab(slabSize int) *slab {
	return &slab{
		mem: make([]byte, slabSize),
	}
}

func (s *slab) get(size int) []byte {
	if s.pos+size > len(s.mem) {
		s.pos = 0
	}
	buf := s.mem[s.pos : s.pos+size : s.pos+size] // s.mem[low : high : max], full-sliced, no capacity bleed
	s.pos += size
	return buf
}

func processArenaPool(arenaPool *slab) int {
	data := arenaPool.get(1024)

	sum := 0
	for i := 0; i < len(data); i++ {
		data[i] = byte(i % 256)
		sum += int(data[i])
	}
	return sum
}

func runArenaPool(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	pool := newSlab(1024)

	// Measure latency
	var minLatency, maxLatency, totalLatency time.Duration

	for i := 0; i < b.N; i++ {
		start := time.Now()
		_ = processArenaPool(pool)
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
