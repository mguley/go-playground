package processor

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// OptimizedConcurrentProcessor fixes concurrency issues
type OptimizedConcurrentProcessor struct {
	workers   int
	processor *OptimizedProcessor

	// Buffered channel to reduce blocking
	jobQueue chan *Document

	// Worker pool management
	workerPool sync.Pool
	cancel     context.CancelFunc
	done       chan struct{}

	// Atomic stats to reduce lock contention
	stats OptimizedStats
}

// OptimizedStats uses atomic operations
type OptimizedStats struct {
	TotalProcessed   atomic.Int64
	TotalErrors      atomic.Int64
	TotalProcessTime atomic.Int64 // in nanoseconds
	ActiveWorkers    atomic.Int32
}

// NewOptimizedConcurrentProcessor creates an optimized processor
func NewOptimizedConcurrentProcessor(workers int) *OptimizedConcurrentProcessor {
	return &OptimizedConcurrentProcessor{
		workers:   workers,
		processor: NewOptimizedProcessor(),
		jobQueue:  make(chan *Document, workers*10), // Buffered channel
		done:      make(chan struct{}),
	}
}

// Start begins processing with proper context handling
func (ocp *OptimizedConcurrentProcessor) Start(ctx context.Context) {
	ctx, ocp.cancel = context.WithCancel(ctx)

	// Create a worker pool
	var wg sync.WaitGroup
	for i := 0; i < ocp.workers; i++ {
		wg.Add(1)
		go ocp.worker(ctx, &wg, i)
	}

	// Signal when all workers are done
	go func() {
		wg.Wait()
		close(ocp.done)
	}()
}

// worker processes documents efficiently
func (ocp *OptimizedConcurrentProcessor) worker(ctx context.Context, wg *sync.WaitGroup, id int) {
	defer wg.Done()

	ocp.stats.ActiveWorkers.Add(1)
	defer ocp.stats.ActiveWorkers.Add(-1)

	// Process documents until canceled
	for {
		select {
		case <-ctx.Done():
			return
		case doc, ok := <-ocp.jobQueue:
			if !ok {
				return
			}

			start := time.Now()
			_, err := ocp.processor.ProcessDocument(doc)
			elapsed := time.Since(start)

			// Update stats with atomic operations
			if err != nil {
				ocp.stats.TotalErrors.Add(1)
			} else {
				ocp.stats.TotalProcessed.Add(1)
				ocp.stats.TotalProcessTime.Add(elapsed.Nanoseconds())
			}
		}
	}
}

// Submit adds a document for processing without blocking
func (ocp *OptimizedConcurrentProcessor) Submit(doc *Document) error {
	select {
	case ocp.jobQueue <- doc:
		return nil
	default:
		// Channel full, could return an error or implement backpressure
		return ErrQueueFull
	}
}

// SubmitWithTimeout adds a document with timeout
func (ocp *OptimizedConcurrentProcessor) SubmitWithTimeout(doc *Document, timeout time.Duration) error {
	select {
	case ocp.jobQueue <- doc:
		return nil
	case <-time.After(timeout):
		return ErrSubmitTimeout
	}
}

// GetStats returns current statistics without locking
func (ocp *OptimizedConcurrentProcessor) GetStats() ProcessingStats {
	processed := ocp.stats.TotalProcessed.Load()
	errors := ocp.stats.TotalErrors.Load()
	totalTime := ocp.stats.TotalProcessTime.Load()
	workers := ocp.stats.ActiveWorkers.Load()

	avgTime := time.Duration(0)
	if processed > 0 {
		avgTime = time.Duration(totalTime / processed)
	}

	return ProcessingStats{
		TotalProcessed:   processed,
		TotalErrors:      errors,
		TotalProcessTime: avgTime,
		ActiveWorkers:    workers,
	}
}

// Stop gracefully shuts down the processor
func (ocp *OptimizedConcurrentProcessor) Stop() {
	// Cancel context to signal workers
	if ocp.cancel != nil {
		ocp.cancel()
	}

	// Close the job queue
	close(ocp.jobQueue)

	// Wait for workers to finish
	<-ocp.done
}

var (
	ErrQueueFull     = fmt.Errorf("job queue is full")
	ErrSubmitTimeout = fmt.Errorf("submit timeout")
)
