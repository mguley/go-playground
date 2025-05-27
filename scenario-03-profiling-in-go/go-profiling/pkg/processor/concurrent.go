package processor

import (
	"context"
	"sync"
	"time"
)

// ConcurrentProcessor processes multiple documents concurrently
type ConcurrentProcessor struct {
	workers   int
	processor *DocumentProcessor

	// Intentional issue: An unbuffered channel causing blocking
	jobQueue chan *Document

	wg sync.WaitGroup

	// Intentional issue: Excessive mutex contention
	stats   ProcessingStats
	statsMu sync.Mutex
}

// ProcessingStats tracks processing statistics
type ProcessingStats struct {
	TotalProcessed   int64
	TotalErrors      int64
	TotalProcessTime time.Duration
	ActiveWorkers    int32
}

// NewConcurrentProcessor creates a new concurrent processor
func NewConcurrentProcessor(workers int) *ConcurrentProcessor {
	return &ConcurrentProcessor{
		workers:   workers,
		processor: NewDocumentProcessor(),
		jobQueue:  make(chan *Document), // Unbuffered channel
	}
}

// Start starts the concurrent processor
func (cp *ConcurrentProcessor) Start(ctx context.Context) {
	for i := 0; i < cp.workers; i++ {
		cp.wg.Add(1)

		go cp.worker(ctx, i)
	}
}

// worker processes documents from the job queue
func (cp *ConcurrentProcessor) worker(ctx context.Context, id int) {
	defer cp.wg.Done()

	// Intentional issue: Incrementing stats with high contention
	cp.statsMu.Lock()
	cp.stats.ActiveWorkers++
	cp.statsMu.Unlock()

	defer func() {
		cp.statsMu.Lock()
		cp.stats.ActiveWorkers--
		cp.statsMu.Unlock()
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case doc := <-cp.jobQueue:
			if doc == nil {
				return
			}

			// Process document
			result, err := cp.processor.ProcessDocument(doc)

			// Update stats with lock contention
			cp.statsMu.Lock()
			if err != nil {
				cp.stats.TotalErrors++
			} else {
				cp.stats.TotalProcessed++
				cp.stats.TotalProcessTime += result.ProcessTime
			}
			cp.statsMu.Unlock()

			// Intentional issue: Simulate some blocking operation
			time.Sleep(75 * time.Millisecond)
		}
	}
}

// Submit submits a document for processing
func (cp *ConcurrentProcessor) Submit(doc *Document) {
	// This will block if workers are busy (unbuffered channel)
	cp.jobQueue <- doc
}

// GetStats returns current processing statistics
func (cp *ConcurrentProcessor) GetStats() ProcessingStats {
	cp.statsMu.Lock()
	defer cp.statsMu.Unlock()
	return cp.stats
}

// Stop stops the concurrent processor
func (cp *ConcurrentProcessor) Stop() {
	close(cp.jobQueue)
	cp.wg.Wait()
}
