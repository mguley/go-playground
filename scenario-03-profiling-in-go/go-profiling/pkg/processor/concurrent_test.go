package processor

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"time"
)

func TestGoroutineCleanup(t *testing.T) {
	initialGoroutines := runtime.NumGoroutine()

	// Create and start a processor
	proc := NewOptimizedConcurrentProcessor(10)
	ctx := context.Background()
	proc.Start(ctx)

	// Submit some work
	for i := 0; i < 100; i++ {
		doc := &Document{
			ID:      fmt.Sprintf("test-%d", i),
			Content: "test content",
		}
		if err := proc.Submit(doc); err != nil {
			t.Errorf("Failed to submit document: %v", err)
		}
	}

	// Let it process
	time.Sleep(100 * time.Millisecond)

	// Stop processor
	proc.Stop()

	// Give time for cleanup
	time.Sleep(100 * time.Millisecond)

	// Check goroutine count
	finalGoroutines := runtime.NumGoroutine()
	leaked := finalGoroutines - initialGoroutines

	if leaked > 0 {
		t.Errorf("Goroutine leak detected: %d goroutines leaked", leaked)

		// Debug: print goroutine dump
		buf := make([]byte, 1<<16)
		runtime.Stack(buf, true)
		t.Logf("Goroutine dump:\n%s", buf)
	}
}

func TestConcurrentProcessorLoad(t *testing.T) {
	proc := NewOptimizedConcurrentProcessor(5)
	ctx := context.Background()
	proc.Start(ctx)
	defer proc.Stop()

	// Simulate high load
	start := time.Now()
	submitted := 0
	rejected := 0

	for i := 0; i < 10_000; i++ {
		doc := &Document{
			ID:      fmt.Sprintf("load-test-%d", i),
			Content: generateTestDocument(100).Content,
		}

		if err := proc.Submit(doc); err != nil {
			rejected++
		} else {
			submitted++
		}
	}

	// Wait for processing to complete
	for proc.GetStats().TotalProcessed < int64(submitted) {
		time.Sleep(10 * time.Millisecond)
	}

	elapsed := time.Since(start)
	stats := proc.GetStats()

	t.Logf("Processed %d documents in %v", stats.TotalProcessed, elapsed)
	t.Logf("Throughput: %.2f docs/sec", float64(stats.TotalProcessed)/elapsed.Seconds())
	t.Logf("Rejected: %d", rejected)
	t.Logf("Average processing time: %v", stats.TotalProcessTime)
}
