package main

import (
	"flag"
	"fmt"
	"gcbench/pkg/metrics"
	"math"
	"math/rand"
	"runtime"
	"runtime/debug"
	"time"
)

var (
	modeFlag       = flag.String("mode", "default", "GC mode: default, tuned, disabled")
	durationFlag   = flag.Int("duration", 30, "Benchmark duration in seconds")
	iterationsFlag = flag.Int("iterations", 1_000_000, "Number of iterations")
)

// Global slice to retain objects and create GC pressure
var retainedObjects = make([]*DataObject, 0, 100_000)

// DataObject represents a more complex allocation target
type DataObject struct {
	ID         int
	Data       []byte        // Large data buffer
	References []*DataObject // References to other objects (creates graph complexity)
	CreatedAt  time.Time
}

// newDataObject creates a new data object with the specified size
func newDataObject(id int, sizeKB int) *DataObject {
	obj := &DataObject{
		ID:         id,
		Data:       make([]byte, sizeKB*1024),
		References: make([]*DataObject, 0, 3),
		CreatedAt:  time.Now(),
	}

	// Fill with some data to ensure it's actually allocated
	for i := 0; i < len(obj.Data); i += 1024 {
		obj.Data[i] = byte(id % 256)
	}

	return obj
}

// doComplexWork simulates CPU-intensive operations
func doComplexWork(iterations int) float64 {
	result := 0.0
	for i := 0; i < iterations; i++ {
		result += math.Sqrt(float64(i * 1_000))
	}
	return result
}

func main() {
	flag.Parse()

	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Mode: %s\n", *modeFlag)

	// Configure GC based on mode
	originalGCPercent := debug.SetGCPercent(100) // Start with default

	switch *modeFlag {
	case "default":
		fmt.Println("Using default GC settings")
	case "tuned":
		debug.SetGCPercent(500) // Less frequent GC
		fmt.Println("Using tuned GC settings (500%)")
	case "disabled":
		debug.SetGCPercent(-1) // Disable GC
		fmt.Println("GC disabled")
	default:
		fmt.Println("Unknown mode, using default GC settings")
	}

	// Print initial memory stats
	metrics.PrintMemStats("Before Benchmark")

	// Run allocation benchmark
	fmt.Println("\nRunning allocation benchmark...")
	runBenchmark(*iterationsFlag, *durationFlag)

	// Free the retained objects to reduce memory pressure
	retainedObjects = nil

	// Restore original GC settings and force collection
	if *modeFlag == "disabled" {
		fmt.Println("\nRe-enabling GC for cleanup...")
		debug.SetGCPercent(originalGCPercent)
	}
	runtime.GC()

	// Print final memory stats
	metrics.PrintMemStats("After Benchmark")
}

func runBenchmark(maxIterations, maxSeconds int) {
	recorder := metrics.NewLatencyRecorder()
	deadline := time.Now().Add(time.Duration(maxSeconds) * time.Second)

	// Pre-allocate some objects to reference
	baseObjects := make([]*DataObject, 0, 5)
	for i := 0; i < 5; i++ {
		baseObjects = append(baseObjects, newDataObject(i, 5))
	}

	// Allocations to benchmark
	for i := 0; i < maxIterations && time.Now().Before(deadline); i++ {
		recorder.RecordFunc(func() {
			// Create a more complex object
			obj := newDataObject(i, 10) // 10KB object

			// Add references to existing objects (creates a more complex object graph)
			if len(baseObjects) > 0 {
				// Link to 1-3 random baseObjects to create a more complex graph for GC
				numRefs := rand.Intn(3) + 1
				for j := 0; j < numRefs; j++ {
					idx := rand.Intn(len(baseObjects))
					obj.References = append(obj.References, baseObjects[idx])
				}
			}

			// Do some CPU-intensive work
			_ = doComplexWork(1_000)

			// Retain about 5% of objects to create long-lived heap pressure
			// This forces the GC to do more work when it runs
			if rand.Float64() < 0.05 {
				retainedObjects = append(retainedObjects, obj)

				// If we have too many retained objects, remove an old one
				// This creates a mix of young and old objects
				if len(retainedObjects) > 10_000 {
					idx := rand.Intn(len(retainedObjects) - 1_000)
					retainedObjects = append(retainedObjects[:idx], retainedObjects[idx+1:]...)
				}
			}
		})

		// Print progress every 300,000 iterations
		if i > 0 && i%300_000 == 0 {
			fmt.Printf("Completed %d iterations...\n", i)

			// Print memory stats occasionally to show memory growth
			if i%300_000 == 0 {
				metrics.PrintMemStats(fmt.Sprintf("After %d iterations", i))
			}
		}
	}

	recorder.PrintStats("Allocation Benchmark")
}
