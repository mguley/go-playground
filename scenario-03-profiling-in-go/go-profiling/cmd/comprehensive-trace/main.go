package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/trace"
	"sync"
	"time"
)

// PerformanceScenario represents different types of performance challenges
type PerformanceScenario struct {
	Name        string
	Description string
	Execute     func(ctx context.Context)
}

// setupScenarios creates different performance scenarios to trace
func setupScenarios() []PerformanceScenario {
	return []PerformanceScenario{
		{
			Name:        "channel_contention",
			Description: "High contention on unbuffered channels",
			Execute:     channelContentionScenario,
		},
		{
			Name:        "lock_contention",
			Description: "Multiple goroutines competing for the same mutex",
			Execute:     lockContentionScenario,
		},
		{
			Name:        "memory_pressure",
			Description: "High allocation rate triggering frequent GC",
			Execute:     memoryPressureScenario,
		},
		{
			Name:        "optimal_concurrency",
			Description: "Well-balanced concurrent processing",
			Execute:     optimalConcurrencyScenario,
		},
	}
}

// channelContentionScenario demonstrates channel blocking issues
func channelContentionScenario(ctx context.Context) {
	ctx, task := trace.NewTask(ctx, "ChannelContentionScenario")
	defer task.End()

	trace.Log(ctx, "scenario", "starting channel contention demonstration")

	// Create an unbuffered channel to force contention
	unbufferedChan := make(chan string)

	var wg sync.WaitGroup

	// Start a single slow consumer
	wg.Add(1)
	go func() {
		defer wg.Done()
		consumerCtx, consumerTask := trace.NewTask(ctx, "SlowConsumer")
		defer consumerTask.End()

		for i := 0; i < 10; i++ {
			trace.Log(consumerCtx, "consumer", fmt.Sprintf("waiting for message %d", i))
			msg := <-unbufferedChan
			trace.Log(consumerCtx, "consumer", fmt.Sprintf("received: %s", msg))

			// Simulate slow processing that creates the bottleneck
			time.Sleep(50 * time.Millisecond)
		}
	}()

	// Start multiple fast producers (these will block)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			producerCtx, producerTask := trace.NewTask(ctx, fmt.Sprintf("Producer_%d", id))
			defer producerTask.End()

			for j := 0; j < 2; j++ {
				msg := fmt.Sprintf("message_%d_%d", id, j)
				trace.Log(producerCtx, "producer", fmt.Sprintf("sending: %s", msg))

				// This will block until the consumer is ready - the blocking will be visible in the trace
				sendStart := time.Now()
				unbufferedChan <- msg
				sendDuration := time.Since(sendStart)

				trace.Log(producerCtx, "producer", fmt.Sprintf("sent: %s (blocked for %v)", msg, sendDuration))
			}
		}(i)
	}

	wg.Wait()
	trace.Log(ctx, "scenario", "channel contention demonstration complete")
}

// lockContentionScenario demonstrates mutex contention
func lockContentionScenario(ctx context.Context) {
	ctx, task := trace.NewTask(ctx, "LockContentionScenario")
	defer task.End()

	trace.Log(ctx, "scenario", "starting lock contention demonstration")

	// Shared resource protected by a mutex
	var mu sync.Mutex
	sharedCounter := 0

	var wg sync.WaitGroup

	// Multiple goroutines competing for the same lock
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			workerCtx, workerTask := trace.NewTask(ctx, fmt.Sprintf("LockWorker_%d", id))
			defer workerTask.End()

			for j := 0; j < 5; j++ {
				trace.Log(workerCtx, "lock", fmt.Sprintf("worker %d attempting lock acquisition %d", id, j))

				lockStart := time.Now()
				mu.Lock()
				lockDuration := time.Since(lockStart)

				trace.Log(workerCtx, "lock", fmt.Sprintf("worker %d acquired lock (waited %v)", id, lockDuration))

				// Hold the lock for varying amounts of time to create contention
				holdTime := time.Duration(10+id*2) * time.Millisecond
				time.Sleep(holdTime)

				sharedCounter++
				trace.Log(workerCtx, "work", fmt.Sprintf("worker %d incremented counter to %d", id, sharedCounter))

				mu.Unlock()
				trace.Log(workerCtx, "lock", fmt.Sprintf("worker %d released lock", id))
			}
		}(i)
	}

	wg.Wait()
	trace.Log(ctx, "scenario", fmt.Sprintf("lock contention demonstration complete, final counter: %d", sharedCounter))
}

// memoryPressureScenario demonstrates GC impact
func memoryPressureScenario(ctx context.Context) {
	ctx, task := trace.NewTask(ctx, "MemoryPressureScenario")
	defer task.End()

	trace.Log(ctx, "scenario", "starting memory pressure demonstration")

	var allocations [][]byte

	// Create memory pressure by allocating and holding onto memory
	for i := 0; i < 100; i++ {
		trace.Log(ctx, "memory", fmt.Sprintf("allocation batch %d", i))

		// Allocate several large slices
		batch := make([][]byte, 10)
		for j := 0; j < 10; j++ {
			batch[j] = make([]byte, 1024*1024) // 1MB each
			// Fill with data
			for k := 0; k < len(batch[j]); k += 4096 {
				batch[j][k] = byte(i + j + k)
			}
		}

		allocations = append(allocations, batch...)

		// Occasionally force GC to see its impact in the trace
		if i%25 == 0 {
			trace.Log(ctx, "gc", "forcing garbage collection")
			runtime.GC()
		}

		// Small delay to spread allocations over time
		time.Sleep(5 * time.Millisecond)
	}

	trace.Log(ctx, "memory", fmt.Sprintf("allocated %d MB total", len(allocations)))

	// Clear allocations to trigger cleanup
	allocations = nil
	runtime.GC()

	trace.Log(ctx, "scenario", "memory pressure demonstration complete")
}

// optimalConcurrencyScenario demonstrates well-balanced processing
func optimalConcurrencyScenario(ctx context.Context) {
	ctx, task := trace.NewTask(ctx, "OptimalConcurrencyScenario")
	defer task.End()

	trace.Log(ctx, "scenario", "starting optimal concurrency demonstration")

	// Use buffered channels to avoid blocking
	workQueue := make(chan int, 20)
	results := make(chan string, 20)

	var wg sync.WaitGroup

	// Start the optimal number of workers (matching CPU cores)
	numWorkers := runtime.NumCPU()
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			workerCtx, workerTask := trace.NewTask(ctx, fmt.Sprintf("OptimalWorker_%d", id))
			defer workerTask.End()

			for work := range workQueue {
				trace.Log(workerCtx, "work", fmt.Sprintf("processing work item %d", work))

				// Simulate balanced CPU work
				result := processWorkItem(work)

				trace.Log(workerCtx, "work", fmt.Sprintf("completed work item %d", work))
				results <- result
			}
		}(i)
	}

	// Producer goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		producerCtx, producerTask := trace.NewTask(ctx, "WorkProducer")
		defer producerTask.End()
		defer close(workQueue)

		for i := 0; i < 50; i++ {
			trace.Log(producerCtx, "produce", fmt.Sprintf("submitting work item %d", i))
			workQueue <- i
		}
	}()

	// Result collector
	wg.Add(1)
	go func() {
		defer wg.Done()
		collectorCtx, collectorTask := trace.NewTask(ctx, "ResultCollector")
		defer collectorTask.End()

		collected := 0
		for result := range results {
			trace.Log(collectorCtx, "collect", fmt.Sprintf("collected result: %s", result))
			collected++

			if collected == 50 {
				close(results)
				break
			}
		}
	}()

	wg.Wait()
	trace.Log(ctx, "scenario", "optimal concurrency demonstration complete")
}

// processWorkItem simulates CPU work
func processWorkItem(work int) string {
	// Simulate some CPU-intensive work
	sum := 0.0
	for i := 0; i < 100_000; i++ {
		sum += float64(work+i) * 1.1
	}
	return fmt.Sprintf("result_%d_%.2f", work, sum)
}

func main() {
	// Create a trace file
	traceFile, err := os.Create("comprehensive_performance.out")
	if err != nil {
		panic(err)
	}
	defer func() { _ = traceFile.Close() }()

	// Start tracing
	fmt.Println("Starting comprehensive performance trace...")
	if err = trace.Start(traceFile); err != nil {
		panic(err)
	}
	defer trace.Stop()

	ctx := context.Background()
	scenarios := setupScenarios()

	// Execute each scenario with clear separation
	for i, scenario := range scenarios {
		fmt.Printf("Executing scenario %d: %s\n", i+1, scenario.Description)

		scenario.Execute(ctx)

		// Add pause between scenarios for visual separation in trace
		time.Sleep(200 * time.Millisecond)
	}

	fmt.Println("Comprehensive trace complete. Analyze with: go tool trace comprehensive_performance.out")
}
