# No Garbage Collection in Go: Performance Benchmarks

## Table of Contents

- [Introduction](#introduction)
- [Understanding Garbage Collection in Go](#understanding-garbage-collection-in-go)
- [Prerequisites](#prerequisites)
- [Step 1: Setting Up the Benchmark Environment](#step-1-setting-up-the-benchmark-environment)
- [Step 2: Baseline Performance with Default GC](#step-2-baseline-performance-with-default-gc)
- [Step 3: Disabling Garbage Collection](#step-3-disabling-garbage-collection)
- [Step 4: Financial Market Data Processor](#step-4-financial-market-data-processor)
- [Step 5: Memory Management Strategies Without GC](#step-5-memory-management-strategies-without-gc)
- [Step 6: Benchmarks and Analysis](#step-6-benchmarks-and-analysis)
- [Step 7: Tuning vs. Disabling: The Middle Ground](#step-7-tuning-vs-disabling-the-middle-ground)
- [Production Considerations](#production-considerations)
- [Conclusion](#conclusion)

#### Introduction

In the world of high-performance computing, predictable latency is often more important than throughput.
A system that processes 1 million requests per second with occasional 50ms pauses can be worse than one that handles 800,000 requests per second with consistent 1ms response times.

Go's garbage collector is remarkably efficient, but it introduces an unavoidable tradeoff: occasional pauses for garbage collection in exchange for automatic memory management.
While these pauses are typically measured in microseconds to low milliseconds, they can still be problematic for ultra-latency-sensitive applications.

In this article, we'll explore a controversial technique: completely disabling Go's garbage collector.

However, as we'll demonstrate, simply disabling garbage collection without implementing proper memory management strategies can actually degrade performance dramatically due to unbounded memory growth.

We'll:
- Implement a financial data processing system
- Compare performance with and without garbage collection
- Demonstrate memory management strategies when GC is disabled
- Analyze comprehensive benchmarks to understand the tradeoffs
- Explore hybrid approaches that balance safety and performance

Before we start: disabling garbage collection is an extreme measure that introduces some complexity and risk.
It should only be considered for specialized applications with strict latency requirements, and even then, only after exhausting other optimization strategies.
Moreover, it must always be paired with proper memory management techniques to be effective.

#### Understanding Garbage Collection in Go

Go's garbage collector has evolved significantly since the language's inception.
The current implementation is a concurrent, tri-color, mark-sweep collector designed to minimize pause times at the expense of some throughput and memory overhead.

Here's a simplified explanation of how it works:

1. **Marking phase**: The GC identifies all reachable objects by traversing the object graph
2. **Sweeping phase**: Memory occupied by unreachable objects is freed
3. **Write barrier**: Ensures correctness during concurrent collection

The GC is triggered when the heap size reaches a threshold determined by the `GOGC` environment variable or the `debug.SetGCPercent()` function.

The default is 100%, meaning GC is triggered when the heap has grown by 100% since the last collection.

In practical terms, this means if you have 10MB of live data after a collection, the next collection will start when the heap reaches about 20MB (the original 10MB plus another 10MB of growth).

When the GC runs, it introduces two types of performance impact:

1. **Direct overhead**: CPU time spent in garbage collection (marking and sweeping)
2. **Pause times**: Brief "stop-the-world" (STW) pauses where all goroutines are suspended

Even with Go's impressive GC optimizations, these impacts can be problematic for applications with strict latency requirements,
especially at the 99.9th percentile and beyond.

#### Prerequisites

To follow along with this article, you'll need:

- Go installed (version 1.24+)
- Basic familiarity with Go programming
- Understanding of memory management concepts
- Benchmarking tools: Go's built-in testing package

#### Step 1: Setting Up the Benchmark Environment

Let's start by creating a benchmark environment that will help us measure the impact of garbage collection.
This benchmark is specifically designed to create significant garbage collection pressure, making the effects of GC clearly visible.

We'll focus on three key metrics:
1. **Operation Latency**: measures how long individual operations take
    - includes mean, median, and percentile breakdowns
    - p99.9 (99.9th percentile) means 99.9% of operations are faster than this value
    - maximum latency reveals the single slowest operation, often representing worst-case scenarios
2. **Memory Usage**: tracks memory consumption patterns
    - heap allocation: active memory used by your program
    - system memory: total memory reserved by Go runtime
    - GC frequency: how often a collection occurs
3. **Throughput**: operations per second your system can handle
    - higher is generally better, but consistency matters more for many applications

When running benchmarks, we need to understand that:
- results will vary significantly based on hardware (CPUs, memory)
- modern processors may show better performance than the article examples
- focus on relative differences between configurations, not absolute numbers

Create a directory structure for our project:

```bash
# Create the project
mkdir -p gc-benchmarks/{cmd,pkg}
cd gc-benchmarks
go mod init gcbench
```

First, let's create utilities to measure and report metrics `pkg/metrics/metrics.go`:

```go
package metrics

import (
   "fmt"
   "math"
   "runtime"
   "sort"
   "time"
)

// LatencyRecorder tracks operation latencies
type LatencyRecorder struct {
   Latencies []time.Duration
   StartTime time.Time
}

// NewLatencyRecorder creates a new latency recorder
func NewLatencyRecorder() *LatencyRecorder {
   return &LatencyRecorder{
      Latencies: make([]time.Duration, 0, 100_000),
      StartTime: time.Now(),
   }
}

// Record adds a latency measurement
func (lr *LatencyRecorder) Record(start time.Time) {
   lr.Latencies = append(lr.Latencies, time.Since(start))
}

// RecordFunc measures and records the execution time of a function
func (lr *LatencyRecorder) RecordFunc(f func()) {
   start := time.Now()
   f()
   lr.Record(start)
}

// PrintStats outputs latency statistics
func (lr *LatencyRecorder) PrintStats(label string) {
   if len(lr.Latencies) == 0 {
      fmt.Printf("%s: No latencies recorded\n", label)
      return
   }
   sort.Slice(lr.Latencies, func(i, j int) bool { return lr.Latencies[i] < lr.Latencies[j] })

   total := len(lr.Latencies)
   var sum time.Duration
   for _, lat := range lr.Latencies {
      sum += lat
   }

   mean := sum / time.Duration(total)
   p50 := percentile(lr.Latencies, 0.50)
   p90 := percentile(lr.Latencies, 0.90)
   p99 := percentile(lr.Latencies, 0.99)
   p999 := percentile(lr.Latencies, 0.999)
   maxLat := lr.Latencies[total-1]

   pureProcessing := sum.Seconds()
   throughput := float64(total) / pureProcessing

   fmt.Printf("\n=== %s ===\n", label)
   fmt.Printf("Operations:  %d\n", total)
   fmt.Printf("Duration:    %v\n", time.Since(lr.StartTime))
   fmt.Printf("Throughput:  %.2f ops/sec\n", throughput)
   fmt.Printf("Mean:        %v\n", mean)
   fmt.Printf("Median (p50): %v\n", p50)
   fmt.Printf("p90:         %v\n", p90)
   fmt.Printf("p99:         %v\n", p99)
   fmt.Printf("p99.9:       %v\n", p999)
   fmt.Printf("Max:         %v\n", maxLat)
}

// MemStats returns current memory statistics
func MemStats() runtime.MemStats {
   var m runtime.MemStats
   runtime.ReadMemStats(&m)
   return m
}

// PrintMemStats outputs memory usage information
func PrintMemStats(label string) {
   var m runtime.MemStats
   runtime.ReadMemStats(&m)

   fmt.Printf("\n=== Memory Stats: %s ===\n", label)
   fmt.Printf("Heap Alloc:   %d MB\n", m.HeapAlloc/1024/1024)
   fmt.Printf("Sys Memory:   %d MB\n", m.Sys/1024/1024)
   fmt.Printf("GC Cycles:    %d\n", m.NumGC)
   if m.NumGC > 0 {
      fmt.Printf("Last GC Pause: %v\n", time.Duration(m.PauseNs[(m.NumGC-1)%256]))
      fmt.Printf("Total GC Pause: %v\n", time.Duration(m.PauseTotalNs))
   }
}

// percentile returns the q‑percentile in [0,1] using linear interpolation.
func percentile(sorted []time.Duration, q float64) time.Duration {
   if len(sorted) == 0 {
      return 0
   }
   if q <= 0 {
      return sorted[0]
   }
   if q >= 1 {
      return sorted[len(sorted)-1]
   }
   pos := q * float64(len(sorted)-1)
   lo := int(math.Floor(pos))
   hi := int(math.Ceil(pos))
   if lo == hi {
      return sorted[lo]
   }
   frac := pos - float64(lo)
   return sorted[lo] + time.Duration(frac*float64(sorted[hi]-sorted[lo]))
}
```

Next, let's create our benchmark runner with a workload that creates significant GC pressure `cmd/benchmark/main.go`:

```go
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
```

This benchmark creates some garbage collection pressure by:
1. **Allocating objects** (10KB) to consume memory faster
2. **Creating complex object graphs** with references between objects, forcing the garbage collector to traverse more complex data structures
3. **Retaining a percentage of objects** (~5%) to create a mix of short and long-lived objects
4. **Performing CPU-intensive work** alongside allocations to compete with the garbage collector for CPU resources
5. **Monitoring memory growth** throughout the benchmark to visualize the impact on heap memory

This benchmark will demonstrate the impact of garbage collection on latency and throughput.

Also, this setup will allow us to run benchmarks with different GC configurations.

#### Step 2: Baseline Performance with Default GC

Before exploring alternative GC strategies, we need to establish a baseline with Go's default garbage collector settings.

This tells us what to expect from Go's standard memory management.

Build and run the benchmark:

```bash
# Build the benchmark tool
go build -o benchmark ./cmd/benchmark

# Run with default GC settings
./benchmark --mode=default --iterations=1_000_000 --duration=30
```

`Understanding Results`

When you run this benchmark with default GC settings, you'll see results similar to:

```
=== Allocation Benchmark ===
Operations:  1000000
Duration:    1.4842941s
Throughput:  754002.68 ops/sec
Mean:        1.326µs
Median (p50): 1.007µs
p90:         1.493µs
p99:         4.13µs
p99.9:       42.917µs
Max:         14.504781ms

=== Memory Stats: After Benchmark ===
Heap Alloc:   0 MB
Sys Memory:   237 MB
GC Cycles:    152
Last GC Pause: 25.705µs
Total GC Pause: 13.174546ms
```

The key observations with default GC settings:

1. **GC Activity**: The benchmark triggered about 152 garbage collection cycles, with pauses totaling about 13ms.
2. **Latency Distribution**:
    - Median operation time is around 1µs
    - 99.9% of operations complete in under 42µs
    - The maximum latency is much higher at around 14.5ms
3. **Memory Usage**: Memory remained well-controlled, with the GC effectively reclaiming memory throughout the run.
4. **Throughput**: The system handles about 750,000 operations per second with these allocations.

The latency profile shows the classic "sawtooth" pattern of garbage collection:
- most operations are relatively fast (p50-p99 range)
- the p99.9 latency is significantly higher (showing rare GC-related pauses)
- Maximum latency (14.5ms) represents major GC events

The significant difference between median latency (1µs) and maximum latency (14.5ms) demonstrates the "stop-the-world" effect of garbage collection.
Even though Go's GC is very efficient, we can observe occasional pauses that affect a small percentage of operations.

`Why This Matters`

The latency distribution is critical for many real-time applications:

- **Web services**: A 14.5ms pause is acceptable for most web applications
- **Gaming servers**: A 14.5ms pause could cause minor "hitching" in fast-paced games
- **Financial trading**: A 14.5ms pause could miss market opportunities and lead to financial losses
- **Real-time bidding**: A 14.5ms pause might exceed bid deadlines, causing missed ad opportunities

While Go's default GC is remarkably efficient for most applications, these pauses could be problematic in ultra-latency sensitive scenarios.

`Hardware considerations`

Your benchmark will likely run on different hardware than what was used in this article.
When comparing your results with the expected values shown here, focus on the patterns and relative differences rather than absolute numbers.

The key principles remain the same regardless of hardware:
1. GC introduces occasional pauses
2. These pauses create outliers in the latency distribution
3. The tradeoff is between memory usage and consistent latency

#### Step 3: Disabling Garbage Collection

Now, let's run the same benchmark with garbage collection disabled.

An important note before proceeding: simply disabling the garbage collector without implementing proper memory management strategies is dangerous and often counterproductive.

As we'll see, this approach can actually degrade performance dramatically due to unbounded memory growth.

```bash
./benchmark --mode=disabled --iterations=1_000_000 --duration=30
```

As you run the benchmark with GC disabled, you'll see significantly different results:

```
=== Allocation Benchmark ===
Operations:  1000000
Duration:    2.996071319s
Throughput:  351736.79 ops/sec
Mean:        2.843µs
Median (p50): 2.474µs
p90:         3.73µs
p99:         11.012µs
p99.9:       41.039µs
Max:         25.902988ms

Re-enabling GC for cleanup...

=== Memory Stats: After Benchmark ===
Heap Alloc:   0 MB
Sys Memory:   9976 MB
GC Cycles:    1
Last GC Pause: 17.321µs
Total GC Pause: 17.321µs
```

The results with disabled GC reveal some critical insights:

1. **Drastically Lower Throughput**: Operations per second dropped from 754,000 to 351,000
2. **Increased Latency**: The median latency more than doubled from 1.00µs to 2.47µs.
3. **Similar p99.9 Latency**: The 99.9th percentile remained almost similar 42.91µs vs. 41.03µs
4. **Variable Maximum Latency**: Maximum latency was sometimes higher and sometimes lower than with GC enabled, showing inconsistent behavior.
5. **Massive Memory Growth**: Memory usage grew from around 237MB to nearly 10GB

This illustrates a critical lesson: simply disabling garbage collection without proper memory management creates more problems than it solves.

The massive memory growth (nearly 10GB for just 1 million objects) creates system-wide performance issues that overwhelm any potential benefits from avoiding GC pauses.

The reduced performance with disabled GC is likely due to:
1. **Memory allocation slowdowns**: As available memory fragments are limited, with many allocations, each new allocation becomes more expensive
2. **CPU cache effects**: With such a large memory footprint, CPU cache utilization becomes inefficient
3. **System memory pressure**: Your operating system might be struggling with the memory demands
4. **Paging/swapping**: The system might be using virtual memory techniques that drastically slow down operations

**Disabling GC without proper memory management makes your program slower, not faster.**

It's not enough to simply turn off the garbage collector - you must also implement strategies to control memory growth.

#### Step 4: Financial Market Data Processor

Now that we've seen the danger of naively disabling garbage collection, let's implement a more realistic example that
incorporates proper memory management: a financial market data processor that ingests, processes, and analyzes market tick data in real-time.

The key difference in this implementation is that we'll use object pooling to reuse memory rather than continuously allocating new objects.

Let's create `pkg/finance/market_data.go`

```go
package finance

import (
   "time"
)

// MarketTick represents a single market price update
type MarketTick struct {
   Symbol    string
   Timestamp time.Time
   Price     float64
   Volume    int
   BidPrice  float64
   AskPrice  float64
   TradeID   int64
}

// MarketStatistics tracks statistics for a symbol
type MarketStatistics struct {
   Symbol      string
   Count       int
   HighPrice   float64
   LowPrice    float64
   VolumeSum   int
   VWAP        float64 // Volume Weighted Average Price
   LastUpdated time.Time
}

// TickProcessor processes market ticks
type TickProcessor struct {
   Stats       map[string]*MarketStatistics
   TickCount   int
   MemoryPool  *TickMemoryPool
   DisablePool bool
}

// NewTickProcessor creates a new tick processor
func NewTickProcessor(usePool bool) *TickProcessor {
   return &TickProcessor{
      Stats:       make(map[string]*MarketStatistics),
      MemoryPool:  NewTickMemoryPool(10_000), // Pre-allocate 10_000 ticks
      DisablePool: !usePool,
   }
}

// GetTick returns a MarketTick (either new or from pool)
func (p *TickProcessor) GetTick() *MarketTick {
   if p.DisablePool {
      return &MarketTick{}
   }
   return p.MemoryPool.Get()
}

// ReleaseTick returns a MarketTick to the pool
func (p *TickProcessor) ReleaseTick(tick *MarketTick) {
   if !p.DisablePool {
      p.MemoryPool.Put(tick)
   }
}

// ProcessTick processes a market tick
func (p *TickProcessor) ProcessTick(tick *MarketTick) {
   p.TickCount++

   // Get or create statistics for this symbol
   stats, exists := p.Stats[tick.Symbol]
   if !exists {
      stats = &MarketStatistics{
         Symbol:    tick.Symbol,
         HighPrice: tick.Price,
         LowPrice:  tick.Price,
      }
      p.Stats[tick.Symbol] = stats
   }

   // Update statistics
   stats.Count++
   stats.LastUpdated = tick.Timestamp

   if tick.Price > stats.HighPrice {
      stats.HighPrice = tick.Price
   }
   if tick.Price < stats.LowPrice {
      stats.LowPrice = tick.Price
   }

   stats.VolumeSum += tick.Volume

   // Update VWAP (Volume Weighted Average Price)
   stats.VWAP = ((stats.VWAP * float64(stats.VolumeSum-tick.Volume)) +
           (tick.Price * float64(tick.Volume))) / float64(stats.VolumeSum)
}

// TickMemoryPool is a simple object pool for MarketTicks
type TickMemoryPool struct {
   pool chan *MarketTick
}

// NewTickMemoryPool creates a new pool with initial capacity
func NewTickMemoryPool(capacity int) *TickMemoryPool {
   pool := &TickMemoryPool{
      pool: make(chan *MarketTick, capacity),
   }

   // Pre-allocate objects
   for i := 0; i < capacity; i++ {
      pool.pool <- &MarketTick{}
   }

   return pool
}

// Get retrieves a MarketTick from the pool or creates a new one
func (p *TickMemoryPool) Get() *MarketTick {
   select {
   case tick := <-p.pool:
      // Reset fields
      tick.Symbol = ""
      tick.Price = 0
      tick.Volume = 0
      tick.BidPrice = 0
      tick.AskPrice = 0
      tick.TradeID = 0
      return tick
   default:
      // Pool exhausted, create a new object
      return &MarketTick{}
   }
}

// Put returns a MarketTick to the pool
func (p *TickMemoryPool) Put(tick *MarketTick) {
   select {
   case p.pool <- tick:
      // Successfully returned to pool
   default:
      // The pool is full
   }
}
```

Now, create a benchmark to test market data processing `cmd/market_benchmark/main.go`:

```go
package main

import (
   "flag"
   "fmt"
   "gcbench/pkg/finance"
   "gcbench/pkg/metrics"
   "math/rand"
   "runtime"
   "runtime/debug"
   "time"
)

var (
   modeFlag     = flag.String("mode", "default", "GC mode: default, tuned, disabled")
   durationFlag = flag.Int("duration", 30, "Benchmark duration in seconds")
   tickRateFlag = flag.Int("tickrate", 100_000, "Market ticks per second to simulate")
   usePoolFlag  = flag.Bool("usepool", false, "Use object pool for MarketTicks")
   symbolsFlag  = flag.Int("symbols", 1_000, "Number of unique symbols to simulate")
)

// Simulates market data
func generateTick(symbols []string) *finance.MarketTick {
   symbolIdx := rand.Intn(len(symbols))
   price := 100.0 + (rand.Float64() * 900.0) // Random price between 100 and 1_000
   volume := rand.Intn(1_000) + 1            // Random volume between 1 and 1_000

   // Small random price fluctuations for bid/ask
   bidPrice := price * (1.0 - (rand.Float64() * 0.001)) // Up to 0.1% lower
   askPrice := price * (1.0 + (rand.Float64() * 0.001)) // Up to 0.1% higher

   return &finance.MarketTick{
      Symbol:    symbols[symbolIdx],
      Timestamp: time.Now(),
      Price:     price,
      Volume:    volume,
      BidPrice:  bidPrice,
      AskPrice:  askPrice,
      TradeID:   rand.Int63(),
   }
}

func main() {
   flag.Parse()

   fmt.Printf("Go Version: %s\n", runtime.Version())
   fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
   fmt.Printf("Mode: %s\n", *modeFlag)
   fmt.Printf("Using object pool: %v\n", *usePoolFlag)

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

   // Generate symbol list
   symbols := make([]string, *symbolsFlag)
   for i := 0; i < *symbolsFlag; i++ {
      symbols[i] = fmt.Sprintf("SYM%04d", i)
   }

   // Print initial memory stats
   metrics.PrintMemStats("Before Benchmark")

   // Create processor
   processor := finance.NewTickProcessor(*usePoolFlag)

   // Run benchmark
   fmt.Printf("\nRunning market data benchmark with %d tick/sec for %d seconds...\n",
      *tickRateFlag, *durationFlag)

   recorder := metrics.NewLatencyRecorder()
   deadline := time.Now().Add(time.Duration(*durationFlag) * time.Second)

   tickInterval := time.Second / time.Duration(*tickRateFlag)
   ticker := time.NewTicker(tickInterval)
   defer ticker.Stop()

   tickCount := 0

   // Use a pre-created tick object if using a pool
   var tick *finance.MarketTick

   for time.Now().Before(deadline) {
      <-ticker.C

      recorder.RecordFunc(func() {
         // Get tick (either new or from pool)
         tick = processor.GetTick()

         // Fill with data
         *tick = *generateTick(symbols)

         // Process the tick
         processor.ProcessTick(tick)

         // Return to the pool if using pooling
         processor.ReleaseTick(tick)
      })

      tickCount++
      if tickCount%1_000_000 == 0 {
         fmt.Printf("Processed %d million ticks...\n", tickCount/1_000_000)
      }
   }

   // Print results
   fmt.Printf("\nProcessed %d market ticks across %d symbols\n",
      processor.TickCount, len(processor.Stats))

   recorder.PrintStats("Market Data Processing")

   // Print a few sample statistics
   fmt.Println("\nSample market statistics:")
   count := 0
   for symbol, stats := range processor.Stats {
      if count < 5 {
         fmt.Printf("%s: Count=%d, High=%.2f, Low=%.2f, Volume=%d, VWAP=%.2f\n",
            symbol, stats.Count, stats.HighPrice, stats.LowPrice,
            stats.VolumeSum, stats.VWAP)
         count++
      } else {
         break
      }
   }

   // If GC was disabled, re-enable it for cleanup
   if *modeFlag == "disabled" {
      fmt.Println("\nRe-enabling GC for cleanup...")
      debug.SetGCPercent(originalGCPercent)
   }
   runtime.GC()

   // Print final memory stats
   metrics.PrintMemStats("After Benchmark")
}
```

Now build and run the market data benchmark:

```bash
go build -o market_benchmark ./cmd/market_benchmark
./market_benchmark --mode=default --tickrate=100_000 --duration=30 --usepool=false
```

This simulates a system processing 100,000 market updates per second for 30 seconds, similar to what a financial trading system might handle.

Importantly, try running with and without object pooling to see the difference:

```bash
# without object pooling
./market_benchmark --mode=default --tickrate=100_000 --duration=30 --usepool=false

# with object pooling
./market_benchmark --mode=default --tickrate=100_000 --duration=30 --usepool=true

# disabled GC without object pooling 
./market_benchmark --mode=disabled --tickrate=100_000 --duration=30 --usepool=false

# disabled GC with object pooling
./market_benchmark --mode=disabled --tickrate=100_000 --duration=30 --usepool=true
```

`Benchmark Comparison Table`

| Configuration           | Default GC  | Default GC + Pool | Disabled GC | Disabled GC + Pool |
|-------------------------|-------------|-------------------|-------------|--------------------|
| **Basic Info**          |             |                   |             |                    |
| Go Version              | go1.24.0    | go1.24.0          | go1.24.0    | go1.24.0           |
| GOMAXPROCS              | 24          | 24                | 24          | 24                 |
| Object Pool             | false       | true              | false       | true               |
| **Performance Metrics** |             |                   |             |                    |
| Operations Processed    | 557,396     | 523,996           | 514,399     | 510,581            |
| Duration (seconds)      | 30.036      | 30.031            | 30.036      | 30.030             |
| Throughput (ops/sec)    | 994,114.40  | 965,977.61        | 880,734.60  | 869,755.46         |
| **Latency (µs)**        |             |                   |             |                    |
| Mean                    | 1,005 µs    | 1,035 µs          | 1,135 µs    | 1,149 µs           |
| Median (p50)            | 671 ns      | 714 ns            | 679 ns      | 723 ns             |
| p90                     | 1,689 µs    | 1,798 µs          | 1,796 µs    | 1,969 µs           |
| p99                     | 5,085 µs    | 4,932 µs          | 10,135 µs   | 7,157 µs           |
| p99.9                   | 21,237 µs   | 19,190 µs         | 30,695 µs   | 28,196 µs          |
| Max                     | 511,827 µs  | 698,055 µs        | 208,318 µs  | 408,006 µs         |
| **Memory Usage**        |             |                   |             |                    |
| Sys Memory Before       | 6 MB        | 6 MB              | 6 MB        | 6 MB               |
| Sys Memory After        | 27 MB       | 36 MB             | 108 MB      | 67 MB              |
| GC Cycles               | 28          | 13                | 1*          | 1*                 |
| Last GC Pause           | 28.294 µs   | 32.863 µs         | 36.288 µs*  | 77.748 µs*         |
| Total GC Pause          | 2.259834 ms | 1.332349 ms       | 36.288 µs*  | 77.748 µs*         |

*Note: For disabled GC configurations, there was only one GC cycle at the end when GC was re-enabled for cleanup.

`Key Observations`:
1. **Throughput**:
    - Default GC showed the highest throughput at ~994,114 ops/sec
    - Disabling GC without object pooling reduced throughput to ~880,734 ops/sec (~11% decrease)
    - Object pooling slightly reduced throughput in both GC and no-GC scenarios
2. **Latency**:
    - Default GC had better mean latency compared to disabled GC
    - Disabled GC showed significantly higher p99 latencies (10,135 µs vs 5,085 µs)
    - Tail latencies (p99.9) were worse with disabled GC
3. **Memory usage**:
    - Disabling GC led to 4x higher memory consumption (108 MB vs. 27 MB)
    - Object pooling helped reduce memory usage in the disabled GC case by ~38% (67 MB vs. 108 MB)
    - Object pooling reduced GC cycles by more than 50% (13 vs. 28)
4. **GC impact**:
    - Object pooling with default GC reduced total GC pause time by ~41% (1.33 ms vs. 2.26 ms)
    - Object pooling consistently improved memory efficiency regardless of GC settings

This table clearly highlights that disabling the GC without proper memory management strategies can lead to worse performance and significantly higher memory usage.

Object pooling shows benefits in reducing GC pressure but doesn't fully compensate for the performance impact of disabling the garbage collector.

#### Step 5: Memory Management Strategies Without GC

In our market data processor, we've implemented a simple object pooling strategy.
In this section, we'll explore a set of memory management techniques that are essential when operating with
disabled GC and show how to apply these principles to create high-performance, memory-efficient Go applications.

`Why memory management matters without GC`

As we saw in the benchmark results from Step 4, disabling garbage collector without proper memory management led to:
1. **4x higher memory consumption** (108MB vs. 27MB)
2. **Reduced throughput** (~11% decrease)
3. **Higher latencies** at p99 and p99.9

The fundamental challenge is clear: without garbage collection, memory will grow unbounded unless we implement manual management strategies.

`Key memory management techniques`

**1. Object pooling**

Object pooling is the practice of pre-allocating and reusing objects rather than creating new ones.

While our initial implementation was simple, production-grade object pools need additional features:

```go
// Enhanced pool with monitoring and auto-sizing
type EnhancedMemoryPool struct {
    pool        chan *MarketTick
    misses      atomic.Int64   // Count of pool misses (had to create new objects)
    hits        atomic.Int64   // Count of pool hits (reused objects)
    maxSize     int            // Maximum pool size
    currentSize atomic.Int64   // Current number of objects in the pool
}

// NewEnhancedMemoryPool creates a pool with monitoring
func NewEnhancedMemoryPool(initialSize, maxSize int) *EnhancedMemoryPool {
    pool := &EnhancedMemoryPool{
        pool:    make(chan *MarketTick, maxSize),
        maxSize: maxSize,
    }
    
    // Pre-allocate objects
    for i := 0; i < initialSize; i++ {
        pool.pool <- &MarketTick{}
    }
    pool.currentSize.Store(int64(initialSize))
    
    return pool
}

// Get retrieves an object with metrics
func (p *EnhancedMemoryPool) Get() *MarketTick {
    select {
    case obj := <-p.pool:
        p.hits.Add(1)
        p.currentSize.Add(-1)
        // Reset object fields
        return obj
    default:
        // Pool exhausted, create a new object
        p.misses.Add(1)
        return &MarketTick{}
    }
}

// Put returns an object to the pool
func (p *EnhancedMemoryPool) Put(obj *MarketTick) {
    // Only add back if we're under capacity
    if p.currentSize.Load() < int64(p.maxSize) {
        select {
        case p.pool <- obj:
            p.currentSize.Add(1)
        default:
            // Pool is full, discard object
        }
    }
}

// GetMetrics returns pool statistics
func (p *EnhancedMemoryPool) GetMetrics() (hits, misses int64, utilization float64) {
    hitsVal := p.hits.Load()
    missesVal := p.misses.Load()
    
    total := hitsVal + missesVal
    utilization = 0
    if total > 0 {
        utilization = float64(hitsVal) / float64(total) * 100.0
    }
    return hitsVal, missesVal, utilization
}
```

Pool implementation options:
- `Channel-based pools`: great for concurrent access, but limited to fixed-size
- `Slice-based pools with mutex`: more flexible sizing, but require careful synchronization
- `Tiered pools`: multiple pools for different object sizes
- `Go's built-in sync.Pool`: convenient and highly optimized (sharded per P), but interacts with GC

`sync.Pool considerations`:

```go
// Using sync.Pool from the standard library
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 1024)
    },
}

func processWithSyncPool() {
    // Get a buffer from the pool
    buffer := bufferPool.Get().([]byte)
    
    // Important: Return to the pool when done
    defer bufferPool.Put(buffer)
    
    // Use the buffer...
}
```

While `sync.Pool` is extremely convenient and well-optimized (sharded per P and integrated with the scheduler), it does interact with the garbage collector.

After each GC cycle, `sync.Pool` drops idle objects to prevent memory leaks.

This makes it ideal for applications that want GC enabled but need to reduce allocation pressure, but less suitable when GC is completely disabled.

Key considerations when implementing object pools:
- pool sizing: too small = too many misses, too large = wasted memory
- reset cost: complex objects may be expensive to reset
- contention: high-concurrency systems need lock-free pools or sharded pools
- atomic counters: use atomic operations instead of mutexes for metrics in high-throughput scenarios
- monitoring: track hit/miss ratios to tune pool sizes

**2. Specialized Buffer Management**

Building on object pooling, we can create specialized buffer pools for operations that need temporary working memory.

This is crucial for avoiding allocations in tight loops, especially for data-processing operations:

```go
// DataBuffer represents a reusable buffer for market analysis
type DataBuffer struct {
    Prices  []float64
    Volumes []int
    Times   []int64 // Unix timestamps
    Size    int
}

// BufferPool manages a pool of pre-allocated data buffers
type BufferPool struct {
    buffers  []*DataBuffer
    mutex    sync.Mutex
    size     int
    capacity int
}

// NewBufferPool creates a buffer pool with specified buffer size and pool capacity
func NewBufferPool(bufferSize, poolCapacity int) *BufferPool {
    pool := &BufferPool{
        buffers:  make([]*DataBuffer, 0, poolCapacity),
        size:     bufferSize,
        capacity: poolCapacity,
    }
    
    // Pre-allocate buffers
    for i := 0; i < poolCapacity; i++ {
        buffer := &DataBuffer{
            Prices:  make([]float64, 0, bufferSize),
            Volumes: make([]int, 0, bufferSize),
            Times:   make([]int64, 0, bufferSize),
        }
        pool.buffers = append(pool.buffers, buffer)
    }
    
    return pool
}

// Get retrieves a buffer from the pool
func (p *BufferPool) Get() *DataBuffer {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    if len(p.buffers) == 0 {
        // Pool exhausted, create a new buffer
        return &DataBuffer{
            Prices:  make([]float64, 0, p.size),
            Volumes: make([]int, 0, p.size),
            Times:   make([]int64, 0, p.size),
        }
    }
    
    // Remove the last buffer
    lastIdx := len(p.buffers) - 1
    buffer := p.buffers[lastIdx]
    p.buffers = p.buffers[:lastIdx]
    
    // Reset buffer
    buffer.Prices = buffer.Prices[:0]
    buffer.Volumes = buffer.Volumes[:0]
    buffer.Times = buffer.Times[:0]
    buffer.Size = 0
    
    return buffer
}

// Put returns a buffer to the pool
func (p *BufferPool) Put(buffer *DataBuffer) {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    // Only add back to the pool if we have capacity
    if len(p.buffers) < p.capacity {
        p.buffers = append(p.buffers, buffer)
    }
}
```

For high-throughput production systems, consider:
- `Sharded pools per goroutine/processor`: Each goroutine or processor gets its own pool, eliminating contention
- `Lock-free ring buffers`: For extremely high-performance scenarios, lock-free data structures can provide better performance

```go
// Example of a goroutine-local buffer pool using goroutine IDs
// Note: This is a simplified example; real implementations need more complexity
type ShardedBufferPool struct {
    pools []*BufferPool
    mask  uint64
}

func NewShardedBufferPool(bufferSize, poolCapacity, shards int) *ShardedBufferPool {
    // Round up to power of 2 for fast modulo
    shardCount := 1
    for shardCount < shards {
        shardCount *= 2
    }
    
    p := &ShardedBufferPool{
        pools: make([]*BufferPool, shardCount),
        mask:  uint64(shardCount - 1),
    }
    
    for i := 0; i < shardCount; i++ {
        p.pools[i] = NewBufferPool(bufferSize, poolCapacity)
    }
    
    return p
}

// getShardIndex returns a consistent shard for the current goroutine
func (p *ShardedBufferPool) getShardIndex() int {
    // This is a simplification. The real implementation would use a method to identify the current goroutine
    return int(p.mask)
}

func (p *ShardedBufferPool) Get() *DataBuffer {
    return p.pools[p.getShardIndex()].Get()
}

func (p *ShardedBufferPool) Put(buffer *DataBuffer) {
    p.pools[p.getShardIndex()].Put(buffer)
}
```

This buffer pool is good for handling the temporary arrays needed for financial calculations like moving averages,
volatility metrics, and other statistical analyses.

**3. Struct Values vs. Pointers**

The choice between passing/returning structs by value vs. by pointer has significant memory implications:

```go
// Returns a pointer (allocates on heap)
func CreateDataPointer() *MarketData {
    return &MarketData{
        Price:  100.0,
        Volume: 1000,
    }
}

// Returns a value (may use stack)
func CreateDataValue() MarketData {
    return MarketData{
        Price:  100.0, 
        Volume: 1000,
    }
}
```

Important to note: size alone doesn't determine stack vs. heap allocation.

The key factor is `escape analysis` - whether the compiler determines that a value might `"escape"` its function scope.

General guidelines:
- small structs: prefer values
- large structs: prefer pointers
- high-modification frequency: prefer pointers
- read-only access: prefer values (immutable)

**4. Pre-allocation**

Allocate with sufficient capacity upfront to avoid expensive reallocations:

```go
// Will cause multiple reallocations
func BuildPriceHistoryBad(symbol string, days int) []float64 {
    prices := make([]float64, 0)  // No capacity hint
    for i := 0; i < days; i++ {
        // Append will trigger reallocation
        prices = append(prices, getPrice(symbol, i))
    }
    return prices
}

// One allocation
func BuildPriceHistoryGood(symbol string, days int) []float64 {
    prices := make([]float64, 0, days)  // Capacity hint
    for i := 0; i < days; i++ {
        // No reallocations needed
        prices = append(prices, getPrice(symbol, i))
    }
    return prices
}
```

This principle applies to all growable data structures: maps, slices, and custom containers.

For maps, use the capacity hint to avoid expensive rehashing operations:

```go
// Will cause progressive rehashing as entries are added
symbolStatsMapBad := make(map[string]*StockStats)

// Pre-allocate buckets for expected capacity
symbolStatsMapGood := make(map[string]*StockStats, 1_000) // Expect 1_000 symbols
```

**5. Arena allocation**

For complex object graphs with the same lifetime, arena allocation can dramatically reduce allocation overhead:

```go
// Simple memory arena
type Arena struct {
    slabs    [][]byte  // Memory slabs
    current  int       // Current slab index
    position int       // Position in the current slab
    slabSize int       // Size of each slab
}

// Create a new arena
func NewArena(slabSize, initialSlabs int) *Arena {
    arena := &Arena{
        slabs:    make([][]byte, initialSlabs),
        slabSize: slabSize,
    }
    
    // Allocate initial slabs
    for i := 0; i < initialSlabs; i++ {
        arena.slabs[i] = make([]byte, slabSize)
    }
    
    return arena
}

// Allocate memory from the arena
func (a *Arena) Allocate(size int) []byte {
    // If it doesn't fit in the current slab, get a new one
    if a.position+size > a.slabSize {
        a.current++
        a.position = 0
        
        // Need a new slab?
        if a.current >= len(a.slabs) {
            a.slabs = append(a.slabs, make([]byte, a.slabSize))
        }
    }
    
    // Allocate from the current slab
    mem := a.slabs[a.current][a.position:a.position+size]
    a.position += size
    return mem
}

// Reset the arena (reuse all memory)
func (a *Arena) Reset() {
    a.current = 0
    a.position = 0
}
```

`Important`: This custom arena implementation is not safe for concurrent goroutines. Only one goroutine should access the arena at a time.

Arenas work especially well for:
- Parsing and processing large datasets
- Temporary object graphs for complex calculations
- Graph algorithms where many nodes are created and then discarded together

`Example`: a "byte-buffer" arena for temporary data

Imagine you want to split a large string into words, copy each word into arena memory, and never worry about freeing them one by one.

```go
package main

import (
	"fmt"
	"unicode"
)

type Arena struct {
	slabs    [][]byte // Memory slabs
	current  int      // Current slab index
	position int      // Position in the current slab
	slabSize int      // Size of each slab
}

func NewArena(slabSize, initialSlabs int) *Arena {
	arena := &Arena{
		slabs:    make([][]byte, initialSlabs),
		slabSize: slabSize,
	}

	// Allocate initial slabs
	for i := 0; i < initialSlabs; i++ {
		arena.slabs[i] = make([]byte, slabSize)
	}

	return arena
}

// Allocate memory from the arena
func (a *Arena) Allocate(size int) []byte {
	// If it doesn't fit in the current slab, get a new one
	if a.position+size > a.slabSize {
		a.current++
		a.position = 0

		// Need a new slab?
		if a.current >= len(a.slabs) {
			a.slabs = append(a.slabs, make([]byte, a.slabSize))
		}
	}

	// Allocate from the current slab
	mem := a.slabs[a.current][a.position : a.position+size]
	a.position += size
	return mem
}

// Reset the arena (reuse all memory)
func (a *Arena) Reset() {
	a.current = 0
	a.position = 0
}

func Tokenize(arena *Arena, text string) []string {
	var tokens []string
	start := 0
	for i, r := range text {
		if unicode.IsSpace(r) {
			if start < i {
				size := i - start
				buf := arena.Allocate(size)
				copy(buf, text[start:i])
				// Convert without re-allocating: share the arena's bytes
				tokens = append(tokens, string(buf))
			}
			start = i + len(string(r))
		}
	}
	// final token
	if start < len(text) {
		size := len(text) - start
		buf := arena.Allocate(size)
		copy(buf, text[start:])
		tokens = append(tokens, string(buf))
	}
	return tokens
}

func main() {
	a := NewArena(1024, 1) // 1 KiB slabs, start with 1 slab
	text := "this is   a test of the arena"
	words := Tokenize(a, text)
	fmt.Println(words) // ["this" "is" "a" "test" "of" "the" "arena"]

	// blow away all allocations
	a.Reset()

	// Re-use the same arena for another command
	more := Tokenize(a, "hello world")
	fmt.Println(more) // ["hello" "world"]
}
```

- Each `Allocate(n)` hands you a `[]byte` slice of length `n` that lives inside the arena's current slab.
- No per-token `make` or `new` calls - everything comes out of the same pool.
- When you call `a.Reset()`, you rewind into slab 0 at position 0. No garbage to collect!

`When to reach for an Arena`
- You're building thousands (or millions) of small objects with the same lifetime.
- You want to avoid the overhead of the garbage collector walking those objects.
- You can free them all at once (no individual frees).

#### Step 6: Benchmarks and Analysis

After exploring various memory management strategies, let's conduct some benchmarks to compare their performance across different garbage collector configurations.

`Benchmark design`

We've designed benchmarks to compare four different memory management approaches:

1. **Direct allocation**: Creating a new buffer for each operation (baseline)
2. **sync.Pool**: Using Go's built-in sync.Pool from the standard library
3. **Channel-based pool**: Using Go channels to manage a pool of reusable buffers
4. **Arena allocation**: Allocating from pre-allocated memory blocks with batch resets

Each approach is tested with three garbage collector configurations:
- **Default GC**: Standard Go garbage collector settings (GOGC = 100)
- **Disabled GC**: Garbage collector completely turned off (GOGC = -1)
- **Tuned GC**: Less frequent garbage collection (GOGC = 500)

This gives us a total of 12 different scenarios to compare.

`What we're measuring`

For each scenario, we measure:

1. **Average Operation Time (ns/op)**: How long each operation takes on average
2. **Mean Operation Time (avg-ns)**: Similar to ns/op but calculated differently
3. **Maximum Latency (max-ns)**: Worst-case operation time
4. **Minimum Latency (min-ns)**: Best-case operation time
5. **Memory Allocations (B/op)**: Bytes allocated per operation
6. **Allocation Count (allocs/op)**: Number of heap allocations per operation

These metrics help us understand both average performance and tail latencies, which are critical for latency-sensitive applications.

`Benchmark implementations`

For consistency, all benchmarks:

- Allocate and work with 1KB buffers
- Perform the same work on the buffer (setting values and calculating a sum)
- Run for 10 seconds with 5 iterations to ensure statistical validity
- Report detailed latency metrics

Let's analyze an example of each memory management approach.

1. **Direct allocation (baseline)**

This is our baseline approach, creating a new buffer for each operation `pkg/benchmark/plain_test.go`:

```go
package benchmark

import (
	"runtime"
	"runtime/debug"
	"testing"
	"time"
	"unsafe"
)

// A global "black-hole" stops the compiler from eliminating work or keeping data on the stack.
var (
	sinkSlice []byte
	_         = unsafe.Sizeof(sinkSlice)
)

//go:noinline
func blackHole(b []byte) {
	sinkSlice = b // force the slice to escape
}

func BenchmarkWithGC(b *testing.B) {
	// Ensure GC is enabled with default settings
	debug.SetGCPercent(100)
	runtime.GC()

	run(b)
}

func BenchmarkWithoutGC(b *testing.B) {
	// Disable GC for benchmark
	originalGCPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	run(b)
}

func BenchmarkWithTunedGC(b *testing.B) {
	// Set a high GC threshold
	originalGCPercent := debug.SetGCPercent(500)
	defer debug.SetGCPercent(originalGCPercent)

	runtime.GC()
	run(b)
}

func allocateAndProcess(size int) int {
	data := make([]byte, size)

	// Do some work with the data
	sum := 0
	for i := 0; i < len(data); i++ {
		data[i] = byte(i % 256)
		sum += int(data[i])
	}

	blackHole(data)
	return sum
}

func run(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	// Measure latency
	var minLatency, maxLatency, totalLatency time.Duration

	for i := 0; i < b.N; i++ {
		start := time.Now()
		_ = allocateAndProcess(1024)
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
```

2. **sync.Pool implementation**

Using Go's built-in sync.Pool to reuse buffers `pkg/benchmark/sync_pool_test.go`:

```go
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
```

3. **Channel-based pool implementation**

Using channels to manage a pool of buffers `pkg/benchmark/channel_pool_test.go`:

```go
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
```

4. **Arena allocation implementation**

Using a pre-allocated memory block and reusing portions of it `pkg/benchmark/arena_pool_test.go`:

```go
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
```

We'll run each benchmark multiple times to ensure statistical validity:

```bash
cd pkg/benchmark
go test -bench=. -benchtime=10s -count=5
```

`Benchmark results`

After running all benchmarks, here are the consolidated results:

| Strategy         | GC Mode     | ns/op  | avg-ns | max-ms | min-ns | B/op | allocs/op |
|------------------|-------------|--------|--------|--------|--------|------|-----------|
| ArenaPool        | Default GC  | 374.0  | 335.2  | 32.1   | 282.6  | 0    | 0         |
| ArenaPool        | Disabled GC | 363.9  | 326.3  | 24.3   | 283.0  | 0    | 0         |
| ArenaPool        | Tuned GC    | 356.2  | 319.1  | 9.24   | 283.0  | 0    | 0         |
| ChanPool         | Default GC  | 432.8  | 396.0  | 17.0   | 342.0  | 0    | 0         |
| ChanPool         | Disabled GC | 435.5  | 398.3  | 19.5   | 342.0  | 0    | 0         |
| ChanPool         | Tuned GC    | 436.8  | 399.6  | 23.5   | 342.0  | 0    | 0         |
| Plain Alloc      | Default GC  | 581.4  | 539.4  | 27.5   | 329.8  | 1024 | 1         |
| Plain Alloc      | Disabled GC | 689.1  | 650.6  | 30.2   | 328.2  | 1024 | 1         |
| Plain Alloc      | Tuned GC    | 552.3  | 513.2  | 26.0   | 330.2  | 1024 | 1         |
| SyncPool         | Default GC  | 480.5  | 442.9  | 15.1   | 333.8  | 24   | 1         |
| SyncPool         | Disabled GC | 480.1  | 442.7  | 18.0   | 334.0  | 24   | 1         |
| SyncPool         | Tuned GC    | 475.1  | 438.1  | 23.0   | 334.0  | 24   | 1         |

`When to use each strategy`

Based on our benchmarks, here are recommendations for when to use each memory management approach:
- **Arena allocation**: Best for ultra-high performance scenarios where objects have similar lifetimes and allocation patterns are predictable.
  Excellent for parsing, data processing, and batch operations.
- **Channel-based pool**: Good for concurrent systems where multiple goroutines need access to shared resources.
  The natural back-pressure from channels can be beneficial in high-throughput scenarios.
- **sync.Pool**: Best general-purpose solution when ease of use is important. It performs well across all GC modes and provides a good balance of performance and convenience.
- **Direct allocation with tuned GC**: When simplicity is paramount and performance is "good enough".
  Tuning the GC rather than disabling it provides better results for typical applications.
- **Disabled GC**: Only recommended in specialized scenarios where you have implemented comprehensive memory management strategies and have thoroughly benchmarked your specific workload.

#### Step 7: Tuning vs. Disabling: The Middle Ground

Based on our benchmarks, completely disabling the GC may be too extreme for most applications.
Let's explore more balanced approaches with a GC controller that gives us fine-grained control `pkg/gccontrol/controller.go`:

```go
package gccontrol

import (
   "context"
   "log"
   "runtime"
   "runtime/debug"
   "sync/atomic"
   "time"
)

// GCController provides advanced control over garbage collection
type GCController struct {
   originalPercent  int           // value returned by debug.SetGCPercent at startup
   forceGCInterval  time.Duration // requested minimum time between forced GCs
   logger           *log.Logger
   disabledCount    atomic.Int32 // number of outstanding DisableGC calls
   lastGCUnixTimeNs atomic.Int64 // updated after every successful runtime.GC
   gcTicker         *time.Ticker
   cancel           context.CancelFunc
}

// Option configures a GCController.
type Option func(*GCController)

// WithLogger lets callers plug in their preferred logger.
func WithLogger(logger *log.Logger) Option { return func(c *GCController) { c.logger = logger } }

// NewGCController creates a new GC controller.
func NewGCController(force time.Duration, opts ...Option) *GCController {
   c := &GCController{
      originalPercent: debug.SetGCPercent(100), // start with default, but remember it
      forceGCInterval: force,
   }
   c.lastGCUnixTimeNs.Store(time.Now().UnixNano())

   for _, opt := range opts {
      opt(c)
   }
   return c
}

// DisableGC increments the disable counter and turns GC off on the first call.
func (c *GCController) DisableGC() {
   if c.disabledCount.Add(1) == 1 { // transition 0 -> 1
      debug.SetGCPercent(-1)
   }
}

// EnableGC decrements the disable counter and re-enables GC once it hits zero.
func (c *GCController) EnableGC() {
   if c.disabledCount.Add(-1) == 0 { // transition 1 -> 0
      debug.SetGCPercent(c.originalPercent)
   }
}

// IsPastForceGCInterval reports whether a forced GC is due.
func (c *GCController) IsPastForceGCInterval() bool {
   last := time.Unix(0, c.lastGCUnixTimeNs.Load())
   return time.Since(last) > c.forceGCInterval
}

// ForceGC temporarily enables GC, runs a collection, and returns to the previous state.
func (c *GCController) ForceGC() {
   wasDisabled := c.disabledCount.Load() > 0

   if wasDisabled {
      // temporarily enable so GC will run
      debug.SetGCPercent(c.originalPercent)
   }

   start := time.Now()
   runtime.GC()
   elapsed := time.Since(start)

   c.lastGCUnixTimeNs.Store(time.Now().UnixNano())

   if c.logger != nil {
      c.logger.Printf("forced GC completed in %s", elapsed)
   }

   if wasDisabled {
      // restore disabled state
      debug.SetGCPercent(-1)
   }
}

// StartScheduledGC launches a goroutine that forces GC every forceGCInterval.
// It can be stopped by calling StopScheduledGC or by cancelling ctx.
func (c *GCController) StartScheduledGC(ctx context.Context) {
   if c.gcTicker != nil {
      return // already running
   }

   ctx, c.cancel = context.WithCancel(ctx)
   c.gcTicker = time.NewTicker(c.forceGCInterval)

   go func(ticker *time.Ticker) {
      defer ticker.Stop()

      for {
         select {
         case <-ticker.C:
            c.ForceGC()
         case <-ctx.Done():
            return
         }
      }
   }(c.gcTicker)
}

// StopScheduledGC cancels the background GC goroutine started by StartScheduledGC.
func (c *GCController) StopScheduledGC() {
   if c.cancel == nil {
      return
   }

   c.cancel()
   c.cancel = nil
}

// DisableGCDuring runs f with GC disabled.
func (c *GCController) DisableGCDuring(f func()) {
   c.DisableGC()
   defer c.EnableGC()
   f()
}

// EnableGCDuring runs f with GC "enabled" even if the controller is currently disabled.
func (c *GCController) EnableGCDuring(f func()) {
   wasDisabled := c.disabledCount.Load() > 0
   if wasDisabled {
      debug.SetGCPercent(c.originalPercent)
   }
   defer func() {
      if wasDisabled {
         debug.SetGCPercent(-1)
      }
   }()
   f()
}
```

Now, let's use our GC controller in the synthetic scenario `cmd/hybrid/main.go`:

```go
package main

import (
   "context"
   "flag"
   "fmt"
   "gcbench/pkg/finance"
   "gcbench/pkg/gccontrol"
   "gcbench/pkg/metrics"
   "log"
   "math/rand"
   "os"
   "os/signal"
   "runtime"
   "syscall"
   "time"
)

var (
   durationFlag = flag.Int("duration", 30, "Benchmark duration in seconds")
   tickRateFlag = flag.Int("tickrate", 100_000, "Market ticks per second to simulate")
   modeFlag     = flag.String("mode", "hybrid", "GC mode: hybrid | disabled | default")
   batchFlag    = flag.Int("batch", 100, "Number of ticks processed per loop iteration")
)

func generateSymbols(n int) []string {
   syms := make([]string, n)
   for i := 0; i < n; i++ {
      syms[i] = fmt.Sprintf("SYM%04d", i)
   }
   return syms
}

func newRand() *rand.Rand {
   return rand.New(rand.NewSource(time.Now().UnixNano()))
}

func main() {
   flag.Parse()

   logger := log.New(os.Stdout, "", log.LstdFlags)
   logger.Printf("Go version : %s", runtime.Version())
   logger.Printf("GOMAXPROCS : %d", runtime.GOMAXPROCS(0))
   logger.Printf("Mode       : %s", *modeFlag)

   // Create GC controller
   var controller *gccontrol.GCController
   switch *modeFlag {
   case "hybrid":
      controller = gccontrol.NewGCController(5*time.Second, gccontrol.WithLogger(logger))
   case "disabled":
      controller = gccontrol.NewGCController(time.Hour, gccontrol.WithLogger(logger))
   default:
      controller = gccontrol.NewGCController(time.Hour) // GC behaves as usual, "default" mode
   }

   ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
   defer stop()

   switch *modeFlag {
   case "hybrid":
      controller.StartScheduledGC(ctx)
      logger.Println("Hybrid GC: GC disabled during batches, forced every 5 s")
   case "disabled":
      controller.DisableGC()
      logger.Println("GC completely disabled")
   }

   // benchmark setup
   rng := newRand()
   // Generate symbol list
   symbols := generateSymbols(1_000)
   // Create a processor with object pooling
   processor := finance.NewTickProcessor(true)
   // Print initial memory stats
   metrics.PrintMemStats("Before benchmark")

   duration := time.Duration(*durationFlag) * time.Second
   deadline := time.Now().Add(duration)
   batchSize := *batchFlag
   tickInt := time.Second / time.Duration(*tickRateFlag)
   batchInt := tickInt * time.Duration(batchSize)
   ticker := time.NewTicker(batchInt)
   defer ticker.Stop()

   recorder := metrics.NewLatencyRecorder()
   var processed int64

   logger.Printf("\nRunning benchmark: %d ticks/sec for %d s (batch=%d)…",
      *tickRateFlag, *durationFlag, batchSize)

loop:
   // Main processing loop
   for {
      select {
      case <-ctx.Done():
         logger.Println("\n⏹  Interrupted – stopping benchmark")
         break loop
      case now := <-ticker.C:
         if now.After(deadline) {
            break loop
         }

         // Process a batch of ticks
         recorder.RecordFunc(func() {
            switch *modeFlag {
            case "hybrid":
               // In hybrid mode, disable GC during batch processing
               controller.DisableGCDuring(func() { processBatch(processor, symbols, batchSize, rng) })
            default:
               processBatch(processor, symbols, batchSize, rng)
            }
         })
         processed += int64(batchSize)

         if processed%(1_000_000) == 0 {
            mil := processed / 1_000_000
            logger.Printf("Processed %d M ticks…", mil)
            metrics.PrintMemStats(fmt.Sprintf("After %d M ticks", mil))
         }
      }
   }

   // results
   logger.Printf("\nProcessed %d market ticks across %d symbols",
      processor.TickCount, len(processor.Stats))

   recorder.PrintStats("Market-data processing")

   // If GC was disabled, re-enable it for cleanup
   switch *modeFlag {
   case "hybrid":
      controller.StopScheduledGC()
   case "disabled":
      controller.EnableGC()
   }

   // Force final GC
   runtime.GC()

   // Print final memory stats
   metrics.PrintMemStats("After benchmark")
}

// process a batch of ticks
func processBatch(processor *finance.TickProcessor, syms []string, n int, rng *rand.Rand) {
   for i := 0; i < n; i++ {
      // Get tick from the pool
      tick := processor.GetTick()

      // Generate random data for the tick
      tick.Symbol = syms[rng.Intn(len(syms))]
      tick.Timestamp = time.Now()
      tick.Price = 100 + rng.Float64()*900
      tick.Volume = rng.Intn(999) + 1
      tick.BidPrice = tick.Price * (1 - rng.Float64()*0.001)
      tick.AskPrice = tick.Price * (1 + rng.Float64()*0.001)
      tick.TradeID = rng.Int63()

      // Process the tick
      processor.ProcessTick(tick)
      // Return to pool
      processor.ReleaseTick(tick)
   }
}
```

Build and run the hybrid mode benchmark:

```bash
go build -o hybrid_benchmark ./cmd/hybrid
./hybrid_benchmark --mode=hybrid --tickrate=100_000 --duration=60
```

This hybrid approach combines the best of both worlds:

1. Uses object pooling to minimize allocation pressure
2. Disables GC during critical processing to minimize latency
3. Periodically enables and forces GC to prevent unbounded memory growth

This approach provides most of the benefits of disabled GC (predictable latency) while avoiding the worst drawbacks (unbounded memory growth).

#### Production Considerations

Before implementing any of these techniques, consider the following

`When to consider disabling GC`

Disabling the GC is only appropriate in very specific scenarios:

1. **Ultra-low** latency requirements: When consistent microsecond-level response times are required
2. **Short-running programs**: Where unbounded memory growth isn't a concern
3. **Bounded memory usage**: When you can guarantee memory won't grow uncontrollably
4. **Critical sections**: During brief periods of intense processing

`Risk Mitigation Strategies`

If you do decide to disable the GC, implement these safeguards:

1. **Memory monitoring**: Track usage continuously and set alerts
2. **Periodic GC**: Re-enable GC periodically to reclaim memory
3. **Memory limits**: Use container or OS-level memory limits
4. **Gradual rollout**: Test extensively before deploying to production
5. **Fallback mechanism**: Have a quick way to revert to standard GC behavior
6. **Memory management**: Always pair disabled GC with proper memory management techniques

`Alternatives to Consider First`

Before disabling the GC, try these less extreme approaches:

1. **Tune GC parameters**: Increase `GOGC` to reduce collection frequency
2. **Object pooling**: Reduce allocation pressure with sync.Pool
3. **Optimize allocations**: Reduce garbage generation at the source
4. **Pre-allocation**: Allocate memory with sufficient capacity upfront
5. **Goroutine pool**: Reuse goroutines instead of creating new ones

#### Conclusion

Disabling Go's garbage collector can provide some improvements in tail latency for ultra-performance sensitive applications,
but only when paired with proper memory management techniques.

Without these techniques, disabling GC actually degrades performance due to unbounded memory growth.

Our benchmarks demonstrate several key insights:

1. **Simply disabling GC is harmful**: It leads to massive memory growth that degrades performance
2. **Object pooling is essential**: It controls memory growth and enables the latency benefits of disabled GC
3. **Hybrid approaches work best**: Combining object pooling with selective GC disabling provides most of the benefits with fewer risks
4. **Tuned GC is often sufficient**: Simply increasing `GOGC` to 500% or higher can provide some latency improvements with minimal risk

For most Go applications, the standard garbage collector provides an excellent balance of performance and convenience.

If you need better latency, start by tuning the GC and optimizing allocations.

Reserve disabling the GC for the small subset of applications where microsecond-level latency consistency is absolutely required,
such as high-frequency trading systems, real-time bidding platforms, or critical infrastructure systems.

Before disabling the garbage collector, always ask: "Is the latency improvement worth the operational complexity and risk?"

If the answer is unclear, stick with Go's standard garbage collection.