package main

import (
	"fmt"
	"os"
	"runtime/trace"
	"sync"
	"time"
)

// simulateWork represents different types of work that might happen in a real application
func simulateWork(workerID int, workType string, duration time.Duration) {
	switch workType {
	case "cpu":
		// Simulate CPU-intensive work
		sum := 0.0
		iterations := int(duration.Nanoseconds() / 1_000) // Scale iterations based on the desired duration
		for i := 0; i < iterations; i++ {
			sum += float64(i) * 1.1
		}
		fmt.Printf("Worker %d completed CPU work: %.2f\n", workerID, sum)

	case "io":
		// Simulate I/O-bound work with sleep
		time.Sleep(duration)
		fmt.Printf("Worker %d completed I/O work\n", workerID)

	case "mixed":
		// Simulate mixed workload
		halfDuration := duration / 2
		time.Sleep(halfDuration) // I/O portion

		// CPU portion
		sum := 0.0
		iterations := int(halfDuration.Nanoseconds() / 2_000)
		for i := 0; i < iterations; i++ {
			sum += float64(i) * 1.1
		}
		fmt.Printf("Worker %d completed mixed work: %.2f\n", workerID, sum)
	}
}

func main() {
	// Create a trace file
	traceFile, err := os.Create("simple_trace.out")
	if err != nil {
		panic(err)
	}
	defer func() { _ = traceFile.Close() }()

	// Start tracing - this begins recording all the events we discussed
	fmt.Println("Starting trace collection...")
	if err = trace.Start(traceFile); err != nil {
		panic(err)
	}
	defer trace.Stop()

	// Create different types of concurrent work to see various patterns in the trace
	var wg sync.WaitGroup

	// Scenario 1: CPU-intensive workers (will show scheduler distribution)
	fmt.Println("Starting CPU-intensive workers...")
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			simulateWork(id, "cpu", 200*time.Millisecond)
		}(i)
	}

	// Wait a moment to see a clear separation in the trace
	time.Sleep(100 * time.Millisecond)

	// Scenario 2: I/O-bound workers (will show blocking behavior)
	fmt.Println("Starting I/O-bound workers...")
	for i := 3; i < 6; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			simulateWork(id, "io", 150*time.Millisecond)
		}(i)
	}

	// Scenario 3: Mixed workload (will show both patterns)
	fmt.Println("Starting mixed workload workers...")
	for i := 6; i < 9; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			simulateWork(id, "mixed", 300*time.Millisecond)
		}(i)
	}

	// Wait for all work to complete
	wg.Wait()
	fmt.Println("All work completed, stopping trace...")
}
