package main

import (
	"context"
	"fmt"
	"goprofiling/pkg/processor"
	"os"
	"runtime/trace"
	"strings"
	"sync"
	"time"
)

// generateTestDocument creates a document for our trace analysis
func generateTestDocument(id int, size int) *processor.Document {
	// Create content of a specified size
	content := strings.Repeat(fmt.Sprintf("test content %d ", id), size/20)

	return &processor.Document{
		ID:      fmt.Sprintf("trace-doc-%d", id),
		Title:   fmt.Sprintf("Trace Test Document %d", id),
		Content: content,
	}
}

// demonstrateChannelBlocking shows the problem with unbuffered channels
func demonstrateChannelBlocking() {
	fmt.Println("=== Demonstrating Channel Blocking Issues ===")

	// Create the problematic processor with unbuffered channels
	proc := processor.NewConcurrentProcessor(3) // Only 3 workers
	ctx := context.Background()
	proc.Start(ctx)
	defer proc.Stop()

	// Try to submit many documents quickly
	// This will cause blocking because the channel is unbuffered
	var submitWg sync.WaitGroup
	documentsToSubmit := 20

	submitStart := time.Now()

	for i := 0; i < documentsToSubmit; i++ {
		submitWg.Add(1)
		go func(docID int) {
			defer submitWg.Done()

			doc := generateTestDocument(docID, 1_000)

			// This Submit call may block due to the unbuffered channel
			blockStart := time.Now()
			proc.Submit(doc)
			blockDuration := time.Since(blockStart)

			// Log any significant blocking
			if blockDuration > 1*time.Millisecond {
				fmt.Printf("Document %d submission blocked for %v\n", docID, blockDuration)
			}
		}(i)
	}

	submitWg.Wait()
	submitDuration := time.Since(submitStart)

	fmt.Printf("Total submission time: %v (average: %v per document)\n",
		submitDuration, submitDuration/time.Duration(documentsToSubmit))
}

// demonstrateOptimizedProcessing shows the improvement with buffered channels
func demonstrateOptimizedProcessing() {
	fmt.Println("\n=== Demonstrating Optimized Processing ===")

	// Create the optimized processor with buffered channels
	proc := processor.NewOptimizedConcurrentProcessor(3) // Same number of workers
	ctx := context.Background()
	proc.Start(ctx)
	defer proc.Stop()

	var submitWg sync.WaitGroup
	documentsToSubmit := 20

	submitStart := time.Now()

	for i := 0; i < documentsToSubmit; i++ {
		submitWg.Add(1)
		go func(docID int) {
			defer submitWg.Done()

			doc := generateTestDocument(docID, 1_000)

			blockStart := time.Now()
			if err := proc.Submit(doc); err != nil {
				fmt.Printf("Document %d submission failed: %v\n", docID, err)
				return
			}
			blockDuration := time.Since(blockStart)

			if blockDuration > 1*time.Millisecond {
				fmt.Printf("Document %d submission blocked for %v\n", docID, blockDuration)
			}
		}(i)
	}

	submitWg.Wait()
	submitDuration := time.Since(submitStart)

	fmt.Printf("Total submission time: %v (average: %v per document)\n",
		submitDuration, submitDuration/time.Duration(documentsToSubmit))
}

func main() {
	// Create a trace file
	traceFile, err := os.Create("concurrency_diagnosis.out")
	if err != nil {
		panic(err)
	}
	defer func() { _ = traceFile.Close() }()

	// Start tracing
	fmt.Println("Starting concurrency diagnosis trace...")
	if err = trace.Start(traceFile); err != nil {
		panic(err)
	}
	defer trace.Stop()

	// First, demonstrate the blocking problem
	demonstrateChannelBlocking()

	// Add a pause to create visual separation in the trace
	time.Sleep(500 * time.Millisecond)

	// Then, demonstrate the optimized solution
	demonstrateOptimizedProcessing()

	// Wait a moment before ending the trace to see final cleanup
	time.Sleep(200 * time.Millisecond)

	fmt.Println("Trace collection complete. Analyze with: go tool trace concurrency_diagnosis.out")
}
