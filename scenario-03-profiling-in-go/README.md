# Profiling in Go: Finding and Fixing Performance Bottlenecks

## Table of Contents
- [Introduction](#introduction)
- [What is Profiling?](#what-is-profiling)
- [Prerequisites](#prerequisites)
- [Step 1: Setting Up Our Profiling Environment](#step-1-setting-up-our-profiling-environment)
- [Step 2: CPU Profiling - Finding Hot Spots](#step-2-cpu-profiling---finding-hot-spots)
- [Step 3: Memory Profiling - Understanding Allocations](#step-3-memory-profiling---understanding-allocations)
- [Step 4: Goroutine and Block Profiling](#step-4-goroutine-and-block-profiling)
- [Step 5: Continuous Profiling with pprof Server](#step-5-continuous-profiling-with-pprof-server)
- [Step 6: Trace Analysis - Understanding Concurrency](#step-6-trace-analysis---understanding-concurrency)
- [Conclusion](#conclusion)

#### Introduction

In production Go applications, performance issues rarely announce themselves clearly. Instead, they manifest as subtle symptoms:

- An API endpoint that becomes sluggish under load
- Memory usage that grows mysteriously over time
- CPU utilization that spikes without an obvious cause
- Goroutines that seem to multiply uncontrollably

For teams building high-performance systems, the ability to diagnose and fix these issues quickly can mean the difference between:

- A search service that responds in 50ms vs. one that takes 500ms
- A data pipeline that processes millions of events per second vs. one that struggles with thousands
- A microservice that handles 10,000 concurrent connections vs. one that crashes at 1,000
- A system that runs smoothly for months vs. one that requires daily restarts

Go provides powerful built-in profiling tools that let you peer inside your running application to understand exactly where time and memory are being spent.
Unlike external monitoring tools that show symptoms, profiling reveals root causes.

In this deep dive, we'll build a simulated content processing system and use Go's profiling tools to identify and fix multiple performance issues.
You'll learn not just how to use these tools, but how to interpret their output and apply fixes that matter.

By the end, you'll have a systematic approach to performance optimization that you can apply to any Go application.

#### What is Profiling?

Profiling is the process of measuring where a program spends its time and memory. Go includes several types of profilers, each designed to answer different questions:

1. **CPU Profile**: Shows which functions consume the most CPU time
    - Answers: "Why is my program slow?"
    - Reveals: Hot spots, inefficient algorithms, excessive computation

2. **Memory Profile**: Shows where memory is being allocated
    - Answers: "Why does my program use so much memory?"
    - Reveals: Memory leaks, excessive allocations, inefficient data structures

3. **Goroutine Profile**: Shows all current goroutines and their stack traces
    - Answers: "Why do I have so many goroutines?"
    - Reveals: Goroutine leaks, deadlocks, concurrency issues

4. **Block Profile**: Shows where goroutines block waiting for synchronization
    - Answers: "Why isn't my concurrent code faster?"
    - Reveals: Lock contention, channel bottlenecks

5. **Mutex Profile**: Shows where goroutines compete for mutexes
    - Answers: "Where is my lock contention?"
    - Reveals: Hot mutexes, opportunities for better concurrency design

6. **Execution Trace**: Records detailed execution events
    - Answers: "What exactly happened during execution?"
    - Reveals: Scheduling latency, GC impact, precise timing

Go's profiling tools are designed to have minimal overhead, making them suitable for use in production environments when configured properly.

#### Prerequisites

Before we begin, you'll need:
- Go installed (version 1.24+)
- Basic understanding of Go concurrency (goroutines, channels, mutexes)
- Familiarity with HTTP servers in Go
- Command-line comfort
- Graphviz installed (optional, for visualizing profiles): [Install instructions](https://graphviz.org/download/)

#### Step 1: Setting Up Our Profiling Environment

Let's create a content processing system that simulates common performance challenges.
This system will process text documents, extract keywords, and serve results via HTTP.

Create our project structure:

```bash
mkdir -p go-profiling/{cmd,pkg,internal}
cd go-profiling
go mod init goprofiling
```

First, let's create our document processor with intentional performance issues that we'll discover and fix through profiling.

Create `pkg/processor/document.go`:

```go
package processor

import (
   "crypto/md5"
   "encoding/hex"
   "fmt"
   "regexp"
   "strings"
   "sync"
   "time"
)

// Document represents a text document to be processed
type Document struct {
   ID        string
   Title     string
   Content   string
   WordCount int
   Keywords  []string
   Hash      string
   Processed time.Time
}

// ProcessingResult contains the results of document processing
type ProcessingResult struct {
   DocID       string
   Keywords    []string
   ProcessTime time.Duration
   MemoryUsed  int64
}

// DocumentProcessor processes documents to extract keywords and metadata
type DocumentProcessor struct {
   processedDocs map[string]*Document
   mu            sync.Mutex

   // Intentional issue: Inefficient regex compilation
   stopWords []string
}

// NewDocumentProcessor creates a new document processor
func NewDocumentProcessor() *DocumentProcessor {
   return &DocumentProcessor{
      processedDocs: make(map[string]*Document),
      stopWords: []string{
         "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
         "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
      },
   }
}

// ProcessDocument processes a single document
func (dp *DocumentProcessor) ProcessDocument(doc *Document) (*ProcessingResult, error) {
   start := time.Now()

   // Intentional issue: Unnecessary string operations
   content := dp.normalizeContent(doc.Content)

   // Intentional issue: Inefficient keyword extraction
   keywords := dp.extractKeywords(content)

   // Intentional issue: Expensive hash calculation
   doc.Hash = dp.calculateHash(doc.Content)

   // Store processed document (memory leak)
   dp.mu.Lock()
   dp.processedDocs[doc.ID] = doc
   dp.mu.Unlock()

   doc.Keywords = keywords
   doc.Processed = time.Now()

   return &ProcessingResult{
      DocID:       doc.ID,
      Keywords:    keywords,
      ProcessTime: time.Since(start),
   }, nil
}

// normalizeContent normalizes the document content
func (dp *DocumentProcessor) normalizeContent(content string) string {
   // Intentional issue: Multiple passes over the same string
   content = strings.ToLower(content)
   content = strings.ReplaceAll(content, "\n", " ")
   content = strings.ReplaceAll(content, "\t", " ")
   content = strings.ReplaceAll(content, ".", " ")
   content = strings.ReplaceAll(content, ",", " ")
   content = strings.ReplaceAll(content, "!", " ")
   content = strings.ReplaceAll(content, "?", " ")
   content = strings.ReplaceAll(content, ";", " ")
   content = strings.ReplaceAll(content, ":", " ")

   // Intentional issue: Regex compiled every time
   re := regexp.MustCompile(`\s+`)
   content = re.ReplaceAllString(content, " ")

   return strings.TrimSpace(content)
}

// extractKeywords extracts keywords from content
func (dp *DocumentProcessor) extractKeywords(content string) []string {
   words := strings.Fields(content)
   wordFreq := make(map[string]int)

   // Intentional issue: Inefficient stop word checking
   for _, word := range words {
      isStopWord := false
      for _, stopWord := range dp.stopWords {
         if word == stopWord {
            isStopWord = true
            break
         }
      }

      if !isStopWord && len(word) > 3 {
         wordFreq[word]++
      }
   }

   // Intentional issue: Inefficient sorting
   var keywords []string
   for word, freq := range wordFreq {
      if freq > 2 {
         // Creating unnecessary string allocations
         keywords = append(keywords, fmt.Sprintf("%s:%d", word, freq))
      }
   }

   return keywords
}

// calculateHash calculates document hash
func (dp *DocumentProcessor) calculateHash(content string) string {
   // Intentional issue: Using MD5 repeatedly on large content
   hasher := md5.New()

   // Intentional issue: Inefficient byte conversion
   for i := 0; i < 10; i++ {
      hasher.Write([]byte(content))
   }

   return hex.EncodeToString(hasher.Sum(nil))
}

// GetProcessedCount returns the number of processed documents
func (dp *DocumentProcessor) GetProcessedCount() int {
   dp.mu.Lock()
   defer dp.mu.Unlock()
   return len(dp.processedDocs)
}
```

Now, let's create a concurrent processor that will reveal goroutine and synchronization issues.

Create `pkg/processor/concurrent.go`:

```go
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
```

Finally, let's create an HTTP server that exposes our processing system and profiling endpoints.

Create `cmd/server/main.go`:

```go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"goprofiling/internal/profiler"
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof" // Import for side effects
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"goprofiling/pkg/processor"
)

var (
	concurrentProc *processor.ConcurrentProcessor
	docCounter     atomic.Int64
)

// generateDocument creates a sample document with random content
func generateDocument() *processor.Document {
	id := docCounter.Add(1)

	// Generate random content of varying sizes
	contentSize := rand.Intn(5_000) + 1_000
	words := []string{
		"performance", "optimization", "golang", "profiling", "memory",
		"cpu", "analysis", "benchmark", "concurrent", "processing",
		"system", "application", "server", "client", "database",
		"network", "algorithm", "structure", "function", "method",
	}

	var contentBuilder strings.Builder
	for i := 0; i < contentSize; i++ {
		contentBuilder.WriteString(words[rand.Intn(len(words))])
		contentBuilder.WriteString(" ")

		if i%10 == 0 {
			contentBuilder.WriteString(". ")
		}
	}

	return &processor.Document{
		ID:      fmt.Sprintf("doc-%d", id),
		Title:   fmt.Sprintf("Document %d", id),
		Content: contentBuilder.String(),
	}
}

// processHandler handles document processing requests
func processHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Generate and process a batch of documents
	batchSize := 10
	var results []map[string]interface{}

	for i := 0; i < batchSize; i++ {
		doc := generateDocument()

		// Submit for processing
		concurrentProc.Submit(doc)

		results = append(results, map[string]interface{}{
			"id":    doc.ID,
			"title": doc.Title,
			"size":  len(doc.Content),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(map[string]interface{}{
		"processed":  results,
		"batch_size": batchSize,
	})
	if err != nil {
		return
	}
}

// statsHandler returns processing statistics
func statsHandler(w http.ResponseWriter, r *http.Request) {
	stats := concurrentProc.GetStats()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := map[string]interface{}{
		"processing": map[string]interface{}{
			"total_processed":  stats.TotalProcessed,
			"total_errors":     stats.TotalErrors,
			"active_workers":   stats.ActiveWorkers,
			"avg_process_time": stats.TotalProcessTime / time.Duration(stats.TotalProcessed+1),
		},
		"memory": map[string]interface{}{
			"alloc":       m.Alloc / 1024 / 1024,      // MB
			"total_alloc": m.TotalAlloc / 1024 / 1024, // MB
			"sys":         m.Sys / 1024 / 1024,        // MB
			"num_gc":      m.NumGC,
			"gc_pause_ns": m.PauseNs[(m.NumGC+255)%256],
		},
		"runtime": map[string]interface{}{
			"goroutines": runtime.NumGoroutine(),
			"cpus":       runtime.NumCPU(),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		return
	}
}

// loadHandler simulates continuous load
func loadHandler(w http.ResponseWriter, r *http.Request) {
	duration := 30 * time.Second
	numGenerators := 15 // concurrent load generators

	// Send an immediate response to the client
	w.Header().Set("Content-Type", "text/plain")
	_, _ = fmt.Fprintf(w, "Load generation started with %d generators for %s\n", numGenerators, duration)

	// Flush the response immediately so the client receives it
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	// Now start the load generation in the background
	var wg sync.WaitGroup

	log.Printf("Starting load generation with %d generators for %s\n", numGenerators, duration)

	for i := 0; i < numGenerators; i++ {
		wg.Add(1)

		go func(id int) {
			defer wg.Done()

			ticker := time.NewTicker(100 * time.Millisecond)
			defer ticker.Stop()

			timeout := time.After(duration)
			documentsSubmitted := 0

			for {
				select {
				case <-timeout:
					log.Printf("Generator %d completed: submitted %d documents\n", id, documentsSubmitted)
					return
				case <-ticker.C:
					// Generate load
					for j := 0; j < 5; j++ {
						doc := generateDocument()

						// Measure blocking time
						start := time.Now()
						concurrentProc.Submit(doc)
						submitDuration := time.Since(start)

						documentsSubmitted++

						// Log occasional blocking measurements
						if submitDuration > 10*time.Millisecond {
							log.Printf("Generator %d: Submit blocked for %s\n", id, submitDuration)
						}
					}
				}
			}
		}(i)
	}

	// Wait for all generators to complete
	go func() {
		wg.Wait()
		log.Printf("All load generators completed after %v\n", duration)
	}()
}

func main() {
	// Enable block profiling
	runtime.SetBlockProfileRate(1)

	// Create a concurrent processor with 10 workers
	concurrentProc = processor.NewConcurrentProcessor(10)

	// Setup context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the processor
	concurrentProc.Start(ctx)

	// Setup continuous profiling
	profilerCfg := profiler.Config{
		OutputDir:   "profiles",
		Interval:    5 * time.Minute,
		CPUDuration: 30 * time.Second,
		EnabledProfiles: []profiler.ProfileType{
			profiler.CPUProfile,
			profiler.HeapProfile,
			profiler.GoroutineProfile,
		},
	}

	continuousProfiler, err := profiler.NewContinuousProfiler(profilerCfg)
	if err != nil {
		log.Fatalf("Failed to create profiler: %v", err)
	}

	continuousProfiler.Start(ctx)
	defer continuousProfiler.Stop()

	// Set up HTTP routes
	http.HandleFunc("/process", processHandler)
	http.HandleFunc("/stats", statsHandler)
	http.HandleFunc("/load", loadHandler)

	// The pprof handlers are automatically registered by importing net/http/pprof
	// They're available at:
	// - /debug/pprof/
	// - /debug/pprof/profile
	// - /debug/pprof/heap
	// - /debug/pprof/goroutine
	// - /debug/pprof/block
	// - /debug/pprof/mutex

	server := &http.Server{
		Addr:         ":8080",
		Handler:      http.DefaultServeMux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 2 * time.Minute,
	}

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("Server starting on %s", server.Addr)
		log.Printf("Profiling endpoints available at http://localhost:8080/debug/pprof/")
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for the termination signal
	<-sigChan
	log.Println("Shutting down server...")

	// Shutdown sequence
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err = server.Shutdown(shutdownCtx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	// Stop the processor
	cancel()
	concurrentProc.Stop()

	log.Println("Server stopped")
}
```

Build and run the server:

```bash
go build -o server ./cmd/server
./server
```

Your profiling environment is now ready. The server exposes:
- `/process` - Process a batch of documents
- `/stats` - View processing statistics
- `/load` - Generate continuous load for testing
- `/debug/pprof/` - Profiling endpoints

This setup includes several intentional performance issues that we'll discover and fix using profiling tools in the following steps.

#### Step 2: CPU Profiling - Finding Hot Spots

CPU profiling helps identify functions that consume the most processing time. Let's use it to find performance bottlenecks in our document processor.

First, let's generate some load and capture a CPU profile:

```bash
# In terminal 1: Generate continuous load
curl http://localhost:8080/load

# In terminal 2: Capture a 30-second CPU profile
go tool pprof -http=:8081 http://localhost:8080/debug/pprof/profile?seconds=30
```

The pprof tool will open a web interface showing the CPU profile. Here's what to look for:

1. **Top Functions**: Shows functions consuming the most CPU time
2. **Graph View**: Visualizes the call graph with hot paths highlighted
3. **Flame Graph**: Shows the stack depth and width representing time spent
4. **Source View**: Shows annotated source code with time percentages

Let's also use the command-line interface for detailed analysis:

```bash
# Generate continuous load
curl http://localhost:8080/load

# Download the profile
curl -o cpu.prof http://localhost:8080/debug/pprof/profile?seconds=30

# Analyze with pprof CLI
go tool pprof cpu.prof

# In the pprof prompt, try these commands:
(pprof) top10                   # Show top 10 functions by CPU time
(pprof) list normalizeContent   # Show source code for a specific function
(pprof) web                     # Generate a graph (requires Graphviz)
```

Based on our intentional issues, you should see hot spots in:
- `normalizeContent` - Multiple string operations and regex compilation
- `extractKeywords` - Inefficient stop word checking
- `calculateHash` - Excessive hashing operations

Let's create an optimized version of our document processor:

Create `pkg/processor/document_optimized.go`:

```go
package processor

import (
   "crypto/md5"
   "encoding/hex"
   "regexp"
   "strings"
   "sync"
   "time"
)

var (
   // Pre-compile regex at the package level
   whitespaceRegex = regexp.MustCompile(`\s+`)

   // Use map for O(1) stop word lookup
   stopWordsMap = map[string]bool{
      "the": true, "a": true, "an": true, "and": true, "or": true,
      "but": true, "in": true, "on": true, "at": true, "to": true,
      "for": true, "of": true, "with": true, "by": true, "from": true,
      "as": true, "is": true, "was": true, "are": true, "were": true,
      "been": true,
   }
)

// OptimizedProcessor is an optimized version of DocumentProcessor
type OptimizedProcessor struct {
   cache *LRUCache

   // Pre-allocated buffers
   bufferPool sync.Pool
}

// NewOptimizedProcessor creates an optimized processor
func NewOptimizedProcessor() *OptimizedProcessor {
   return &OptimizedProcessor{
      cache: NewLRUCache(1_000, 5*time.Minute), // Max 1_000 items, 5 min TTL
      bufferPool: sync.Pool{
         New: func() interface{} {
            return new(strings.Builder)
         },
      },
   }
}

// ProcessDocument processes a document efficiently
func (op *OptimizedProcessor) ProcessDocument(doc *Document) (*ProcessingResult, error) {
   start := time.Now()

   // Use optimized normalization
   content := op.normalizeContentOptimized(doc.Content)

   // Use optimized keyword extraction
   keywords := op.extractKeywordsOptimized(content)

   // Use optimized hash calculation
   doc.Hash = op.calculateHashOptimized(doc.Content)

   // Store in a bounded cache instead of an unbounded map
   op.cache.Put(doc.ID, doc)

   doc.Keywords = keywords
   doc.Processed = time.Now()

   return &ProcessingResult{
      DocID:       doc.ID,
      Keywords:    keywords,
      ProcessTime: time.Since(start),
   }, nil
}

// normalizeContentOptimized efficiently normalizes content
func (op *OptimizedProcessor) normalizeContentOptimized(content string) string {
   // Get a string builder from the pool
   builder := op.bufferPool.Get().(*strings.Builder)
   defer func() {
      builder.Reset()
      op.bufferPool.Put(builder)
   }()

   // Single pass normalization
   builder.Grow(len(content)) // Pre-allocate capacity

   for i, r := range content {
      switch r {
      case '\n', '\t', '.', ',', '!', '?', ';', ':':
         builder.WriteByte(' ')
      default:
         // Convert to lowercase inline
         if r >= 'A' && r <= 'Z' {
            builder.WriteByte(byte(r + 32))
         } else {
            builder.WriteRune(r)
         }
      }
      _ = i // Use index to avoid unused variable warning
   }

   // Use pre-compiled regex
   normalized := whitespaceRegex.ReplaceAllString(builder.String(), " ")
   return strings.TrimSpace(normalized)
}

// extractKeywordsOptimized efficiently extracts keywords
func (op *OptimizedProcessor) extractKeywordsOptimized(content string) []string {
   words := strings.Fields(content)
   wordFreq := make(map[string]int, len(words)/10) // Pre-size map

   // Single pass with map lookup
   for _, word := range words {
      // O(1) stop word check
      if !stopWordsMap[word] && len(word) > 3 {
         wordFreq[word]++
      }
   }

   // Pre-allocate result slice
   keywords := make([]string, 0, len(wordFreq))
   for word, freq := range wordFreq {
      if freq > 2 {
         keywords = append(keywords, word)
      }
   }

   return keywords
}

// calculateHashOptimized efficiently calculates hash
func (op *OptimizedProcessor) calculateHashOptimized(content string) string {
   // Single hash calculation
   hasher := md5.New()
   hasher.Write([]byte(content))
   return hex.EncodeToString(hasher.Sum(nil))
}
```

Create a benchmark to compare the implementations:

Create `pkg/processor/benchmark_test.go`:

```go
package processor

import (
   "fmt"
   "strings"
   "testing"
)

// generateTestDocument creates a test document
func generateTestDocument(size int) *Document {
   var builder strings.Builder
   words := []string{
      "performance", "optimization", "golang", "profiling", "memory",
      "the", "and", "for", "with", "that", "have", "from", "this",
   }

   for i := 0; i < size; i++ {
      builder.WriteString(words[i%len(words)])
      builder.WriteString(" ")
      if i%10 == 0 {
         builder.WriteString(". ")
      }
   }

   return &Document{
      ID:      fmt.Sprintf("test-doc-%d", size),
      Title:   "Test Document",
      Content: builder.String(),
   }
}

func BenchmarkDocumentProcessor(b *testing.B) {
   processor := NewDocumentProcessor()
   doc := generateTestDocument(1_000)

   b.ResetTimer()
   for i := 0; i < b.N; i++ {
      _, _ = processor.ProcessDocument(doc)
   }
}

func BenchmarkOptimizedProcessor(b *testing.B) {
   processor := NewOptimizedProcessor()
   doc := generateTestDocument(1_000)

   b.ResetTimer()
   for i := 0; i < b.N; i++ {
      _, _ = processor.ProcessDocument(doc)
   }
}

// Benchmark specific operations
func BenchmarkNormalizeContent(b *testing.B) {
   processor := NewDocumentProcessor()
   content := generateTestDocument(1_000).Content

   b.ResetTimer()
   for i := 0; i < b.N; i++ {
      _ = processor.normalizeContent(content)
   }
}

func BenchmarkNormalizeContentOptimized(b *testing.B) {
   processor := NewOptimizedProcessor()
   content := generateTestDocument(1_000).Content

   b.ResetTimer()
   for i := 0; i < b.N; i++ {
      _ = processor.normalizeContentOptimized(content)
   }
}
```

Run the benchmarks to see the improvement:

```bash
cd pkg/processor
go test -bench=. -benchmem -cpuprofile=cpu.prof

# View the profile
go tool pprof -http=:8082 cpu.prof
```

The optimizations we applied:

1. **Pre-compiled regex**: Moved regex compilation to package initialization
2. **Single-pass normalization**: Combined multiple string operations into one pass
3. **Map-based stop words**: O(1) lookup instead of O(n) linear search
4. **Pre-allocated buffers**: Reused string builders from a pool
5. **Efficient hashing**: Removed unnecessary repeated hashing

These changes typically result in significant performance improvement for document processing.

#### Step 3: Memory Profiling - Understanding Allocations

Memory profiling helps identify where your program allocates memory and potential memory leaks. Let's analyze our system's memory usage.

First, generate load and capture heap profiles:

```bash
# Generate load
curl http://localhost:8080/load

# Capture heap profile
curl -o heap.prof http://localhost:8080/debug/pprof/heap?seconds=30

# Analyze heap profile
go tool pprof heap.prof
```

In the pprof prompt, useful commands for memory analysis:

```bash
(pprof) top10                 # Top 10 memory allocators
(pprof) list ProcessDocument  # Show source of specific allocation
```

Let's create a tool to detect memory leaks by comparing heap profiles:

Create `cmd/memcheck/main.go`:

```go
package main

import (
   "fmt"
   "io"
   "log"
   "net/http"
   "os"
)

func captureHeapProfile(filename string) error {
   resp, err := http.Get("http://localhost:8080/debug/pprof/heap?seconds=25")
   if err != nil {
      return err
   }
   defer func(Body io.ReadCloser) {
      if err = Body.Close(); err != nil {
         log.Fatal(err)
      }
   }(resp.Body)

   file, err := os.Create(filename)
   if err != nil {
      return err
   }
   defer func(file *os.File) {
      if err = file.Close(); err != nil {
         log.Fatal(err)
      }
   }(file)

   _, err = file.ReadFrom(resp.Body)
   return err
}

func main() {
   // Capture initial heap profile
   fmt.Println("Capturing initial heap profile...")
   if err := captureHeapProfile("heap1.prof"); err != nil {
      log.Fatal(err)
   }

   // Generate load
   fmt.Println("Generating load for 30 seconds...")
   if _, err := http.Get("http://localhost:8080/load"); err != nil {
      return
   }

   // Capture second heap profile
   fmt.Println("Capturing second heap profile...")
   if err := captureHeapProfile("heap2.prof"); err != nil {
      log.Fatal(err)
   }

   fmt.Println("\nCompare profiles with:")
   fmt.Println("go tool pprof -base=heap1.prof heap2.prof")
   fmt.Println("\nThen use 'top' command to see memory growth")
}
```

Run the memory leak detector:

```bash
go run cmd/memcheck/main.go

# Compare the profiles
go tool pprof -base=heap1.prof heap2.prof
(pprof) top
(pprof) list ProcessDocument
```

You might see that the `ProcessDocument` function consumes some memory in different places.

Let's try to fix the memory leak with a bounded cache:

Create `pkg/processor/cache.go`:

```go
package processor

import (
   "container/list"
   "sync"
   "time"
)

// LRUCache implements a thread-safe LRU cache with TTL
type LRUCache struct {
   capacity int
   ttl      time.Duration
   items    map[string]*list.Element
   order    *list.List
   mu       sync.Mutex
}

type cacheEntry struct {
   key       string
   value     *Document
   timestamp time.Time
}

// NewLRUCache creates a new LRU cache
func NewLRUCache(capacity int, ttl time.Duration) *LRUCache {
   return &LRUCache{
      capacity: capacity,
      ttl:      ttl,
      items:    make(map[string]*list.Element),
      order:    list.New(),
   }
}

// Put adds or updates an item in the cache
func (c *LRUCache) Put(key string, value *Document) {
   c.mu.Lock()
   defer c.mu.Unlock()

   // Check if item exists
   if elem, exists := c.items[key]; exists {
      // Move to the front and update
      c.order.MoveToFront(elem)
      elem.Value.(*cacheEntry).value = value
      elem.Value.(*cacheEntry).timestamp = time.Now()
      return
   }

   // Add new item
   entry := &cacheEntry{
      key:       key,
      value:     value,
      timestamp: time.Now(),
   }

   elem := c.order.PushFront(entry)
   c.items[key] = elem

   // Evict if over capacity
   if c.order.Len() > c.capacity {
      c.evictOldest()
   }
}

// Get retrieves an item from the cache
func (c *LRUCache) Get(key string) (*Document, bool) {
   c.mu.Lock()
   defer c.mu.Unlock()

   elem, exists := c.items[key]
   if !exists {
      return nil, false
   }

   entry := elem.Value.(*cacheEntry)

   // Check TTL
   if time.Since(entry.timestamp) > c.ttl {
      c.removeElement(elem)
      return nil, false
   }

   // Move to the front
   c.order.MoveToFront(elem)
   return entry.value, true
}

// evictOldest removes the least recently used item
func (c *LRUCache) evictOldest() {
   elem := c.order.Back()
   if elem != nil {
      c.removeElement(elem)
   }
}

// removeElement removes an element from the cache
func (c *LRUCache) removeElement(elem *list.Element) {
   c.order.Remove(elem)
   entry := elem.Value.(*cacheEntry)
   delete(c.items, entry.key)
}

// Size returns the current size of the cache
func (c *LRUCache) Size() int {
   c.mu.Lock()
   defer c.mu.Unlock()
   return c.order.Len()
}

// Clear removes all items from the cache
func (c *LRUCache) Clear() {
   c.mu.Lock()
   defer c.mu.Unlock()

   c.items = make(map[string]*list.Element)
   c.order = list.New()
}
```

Update the processor to use the bounded cache `pkg/processor/document_optimized.go`:

```go
type OptimizedProcessor struct {
	cache      *LRUCache
	bufferPool sync.Pool
}

func NewOptimizedProcessor() *OptimizedProcessor {
	return &OptimizedProcessor{
		cache: NewLRUCache(1_000, 5*time.Minute), // Max 1_000 items, 5 min TTL
		bufferPool: sync.Pool{
			New: func() interface{} {
				return new(strings.Builder)
			},
		},
	}
}

// Update ProcessDocument to use cache
func (op *OptimizedProcessor) ProcessDocument(doc *Document) (*ProcessingResult, error) {
	// ... processing logic ...
	
	// Store in a bounded cache instead of an unbounded map (original document - pkg/processor/document.go)
	op.cache.Put(doc.ID, doc)
	
	// ... rest of the method ...
}
```

Memory profiling best practices:

1. **Compare profiles**: Always compare before/after profiles to spot growth
2. **Check both alloc and inuse**: Allocations show total memory pressure, inuse shows leaks
3. **Look for unexpected retentions**: Large maps/slices that grow without bounds
4. **Profile regularly**: Memory leaks often develop over time
5. **Use -base flag**: Makes it easy to see what changed between profiles

#### Step 4: Goroutine and Block Profiling

Goroutine leaks and synchronization issues can severely impact performance. Let's analyze our concurrent processor.

Check the current goroutine count:

```bash
# View goroutine profile in browser
go tool pprof -http=:8083 http://localhost:8080/debug/pprof/goroutine

# Or download and analyze
curl -o goroutine.prof http://localhost:8080/debug/pprof/goroutine
go tool pprof goroutine.prof
(pprof) top
```

To identify blocking issues, we need to enable block profiling:

Update `cmd/server/main.go` to enable block profiling:

```go
import (
	"runtime"
	// ... other imports
)

func main() {
	// Enable block profiling
	runtime.SetBlockProfileRate(1)
	
	// ... rest of main function
}
```

After restarting the server and generating load:

```bash
# Rebuild
go build -o server ./cmd/server

# Rerun the server with enabled block profiling
./server

# Generate load
curl http://localhost:8080/load

# Capture block profile
curl -o block.prof http://localhost:8080/debug/pprof/block?seconds=30

# Analyze blocking
go tool pprof block.prof
(pprof) top
(pprof) list Submit
```

You'll likely see blocking on the unbuffered channel in `Submit`. The initial `Submit` function was implemented here - `pkg/processor/concurrent.go`.

Let's create an improved concurrent processor:

Create `pkg/processor/concurrent_optimized.go`:

```go
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
```

Create a test to verify goroutine cleanup:

Create `pkg/processor/concurrent_test.go`:

```go
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
```

Running and understanding the concurrent tests:

Before running the tests, it's important to understand what each test validates:

1. **TestGoroutineCleanup**: This test ensures our concurrent processor doesn't leak goroutines when it shuts down.
   Goroutine leaks are a common source of memory leaks in Go applications, as leaked goroutines can hold references to memory that can never be garbage collected.
2. **TestConcurrentProcessorLoad**: This test validates that our processor can handle high-throughput scenarios while
   maintaining stability and providing meaningful performance metrics.

Navigate to the processor package directory and run the tests:

```bash
cd pkg/processor

# Run all tests in the package
go test -v

# Run tests with race detection (important for concurrent code)
go test -v -race -run="TestConcurrent|TestGoroutine"

# Run tests with memory and CPU profiling
go test -v -cpuprofile=test_cpu.prof -memprofile=test_mem.prof -run="TestConcurrent|TestGoroutine"
```

The improvements we made `pkg/processor/concurrent.go` vs. `pkg/processor/concurrent_optimized.go`:

1. **Buffered channels**: Reduced blocking on job submission
2. **Atomic operations**: Eliminated lock contention on stats
3. **Proper cleanup**: Ensured all goroutines terminate gracefully
4. **Context handling**: Clean cancellation propagation
5. **Backpressure**: Handle queue full scenarios

These changes typically improve throughput and eliminate goroutine leaks.

#### Step 5: Continuous Profiling with pprof Server

For production systems, continuous profiling helps catch performance regressions early. Let's add a profiling server with automated profile collection.

Create `internal/profiler/continuous.go`:

```go
package profiler

import (
   "context"
   "fmt"
   "log"
   "os"
   "path/filepath"
   "runtime"
   "runtime/pprof"
   "runtime/trace"
   "sync"
   "time"
)

// ContinuousProfiler collects profiles at regular intervals
type ContinuousProfiler struct {
   outputDir       string
   interval        time.Duration
   cpuDuration     time.Duration
   enabledProfiles []ProfileType
   logger          *log.Logger

   mu     sync.Mutex
   cancel context.CancelFunc
   done   chan struct{}
}

// ProfileType represents different profile types
type ProfileType string

const (
   CPUProfile       ProfileType = "cpu"
   HeapProfile      ProfileType = "heap"
   GoroutineProfile ProfileType = "goroutine"
   AllocsProfile    ProfileType = "allocs"
   BlockProfile     ProfileType = "block"
   MutexProfile     ProfileType = "mutex"
   TraceProfile     ProfileType = "trace"
)

// Config holds profiler configuration
type Config struct {
   OutputDir       string
   Interval        time.Duration
   CPUDuration     time.Duration
   EnabledProfiles []ProfileType
}

// NewContinuousProfiler creates a new continuous profiler
func NewContinuousProfiler(cfg Config) (*ContinuousProfiler, error) {
   // Create an output directory
   if err := os.MkdirAll(cfg.OutputDir, 0755); err != nil {
      return nil, fmt.Errorf("failed to create output dir: %w", err)
   }

   logger := log.New(os.Stdout, "[profiler] ", log.LstdFlags)

   return &ContinuousProfiler{
      outputDir:       cfg.OutputDir,
      interval:        cfg.Interval,
      cpuDuration:     cfg.CPUDuration,
      enabledProfiles: cfg.EnabledProfiles,
      logger:          logger,
      done:            make(chan struct{}),
   }, nil
}

// Start begins continuous profiling
func (cp *ContinuousProfiler) Start(ctx context.Context) {
   cp.mu.Lock()
   defer cp.mu.Unlock()

   ctx, cp.cancel = context.WithCancel(ctx)

   go cp.run(ctx)

   cp.logger.Printf("Started continuous profiling, interval=%v, output=%s",
      cp.interval, cp.outputDir)
}

// run is the main profiling loop
func (cp *ContinuousProfiler) run(ctx context.Context) {
   defer close(cp.done)

   ticker := time.NewTicker(cp.interval)
   defer ticker.Stop()

   // Collect initial profiles
   cp.collectProfiles()

   for {
      select {
      case <-ctx.Done():
         return
      case <-ticker.C:
         cp.collectProfiles()
      }
   }
}

// collectProfiles collects all enabled profiles
func (cp *ContinuousProfiler) collectProfiles() {
   timestamp := time.Now().Format("20060102-150405")

   for _, profileType := range cp.enabledProfiles {
      switch profileType {
      // CPU profile
      case CPUProfile:
         go cp.collectCPUProfile(timestamp)
      // Memory profiles
      case HeapProfile:
         cp.collectProfile(HeapProfile, timestamp)
      case AllocsProfile:
         cp.collectProfile(AllocsProfile, timestamp)
      // Runtime profiles
      case GoroutineProfile:
         cp.collectProfile(GoroutineProfile, timestamp)
      case BlockProfile:
         cp.collectProfile(BlockProfile, timestamp)
      case MutexProfile:
         cp.collectProfile(MutexProfile, timestamp)
      // Execution trace
      case TraceProfile:
         go cp.collectTrace(timestamp)
      }
   }

   // Always log memory stats
   cp.logMemoryStats(timestamp)
}

// collectCPUProfile collects CPU profile over duration
func (cp *ContinuousProfiler) collectCPUProfile(timestamp string) {
   filename := cp.profileFilename(CPUProfile, timestamp)

   file, err := os.Create(filename)
   if err != nil {
      cp.logger.Printf("Failed to create CPU profile: %v", err)
      return
   }
   defer func() {
      if closeErr := file.Close(); closeErr != nil {
         cp.logger.Printf("Failed to close CPU profile file: %v", closeErr)
      }
   }()

   if err = pprof.StartCPUProfile(file); err != nil {
      cp.logger.Printf("Failed to start CPU profile: %v", err)
      return
   }

   time.Sleep(cp.cpuDuration)
   pprof.StopCPUProfile()

   cp.logger.Printf("Saved CPU profile: %s", filename)
}

// collectProfile collects a runtime profile
func (cp *ContinuousProfiler) collectProfile(profileType ProfileType, timestamp string) {
   var p *pprof.Profile

   switch profileType {
   case HeapProfile:
      p = pprof.Lookup("heap")
   case AllocsProfile:
      p = pprof.Lookup("allocs")
   case GoroutineProfile:
      p = pprof.Lookup("goroutine")
   case BlockProfile:
      p = pprof.Lookup("block")
   case MutexProfile:
      p = pprof.Lookup("mutex")
   default:
      return
   }

   if p == nil {
      return
   }

   filename := cp.profileFilename(profileType, timestamp)
   file, err := os.Create(filename)
   if err != nil {
      cp.logger.Printf("Failed to create %s profile: %v", profileType, err)
      return
   }
   defer func() {
      if closeErr := file.Close(); closeErr != nil {
         cp.logger.Printf("Failed to close %s profile file: %v", profileType, closeErr)
      }
   }()

   if err = p.WriteTo(file, 0); err != nil {
      cp.logger.Printf("Failed to write %s profile: %v", profileType, err)
      return
   }

   cp.logger.Printf("Saved %s profile: %s", profileType, filename)
}

// collectTrace collects execution trace
func (cp *ContinuousProfiler) collectTrace(timestamp string) {
   filename := cp.profileFilename(TraceProfile, timestamp)

   file, err := os.Create(filename)
   if err != nil {
      cp.logger.Printf("Failed to create trace file: %v", err)
      return
   }
   defer func() {
      if closeErr := file.Close(); closeErr != nil {
         cp.logger.Printf("Failed to close trace file: %v", closeErr)
      }
   }()

   if err = trace.Start(file); err != nil {
      cp.logger.Printf("Failed to start trace: %v", err)
      return
   }

   time.Sleep(cp.cpuDuration)
   trace.Stop()

   cp.logger.Printf("Saved trace: %s", filename)
}

// logMemoryStats logs current memory statistics
func (cp *ContinuousProfiler) logMemoryStats(timestamp string) {
   var m runtime.MemStats
   runtime.ReadMemStats(&m)

   cp.logger.Printf("Memory stats at %s: Alloc=%dMB, TotalAlloc=%dMB, Sys=%dMB, NumGC=%d",
      timestamp,
      m.Alloc/1024/1024,
      m.TotalAlloc/1024/1024,
      m.Sys/1024/1024,
      m.NumGC,
   )
}

// profileFilename generates a filename for a profile
func (cp *ContinuousProfiler) profileFilename(profileType ProfileType, timestamp string) string {
   return filepath.Join(cp.outputDir, fmt.Sprintf("%s_%s.prof", profileType, timestamp))
}

// Stop stops the continuous profiler
func (cp *ContinuousProfiler) Stop() {
   cp.mu.Lock()
   if cp.cancel != nil {
      cp.cancel()
   }
   cp.mu.Unlock()

   <-cp.done
   cp.logger.Println("Stopped continuous profiling")
}
```

Update the server to use continuous profiling in `cmd/server/main.go`:

```go
import (
	"goprofiling/internal/profiler"
	// ... other imports
)

func main() {
	// ... existing setup ...
	
	// Setup continuous profiling
	profilerCfg := profiler.Config{
		OutputDir:    "profiles",
		Interval:     5 * time.Minute,
		CPUDuration:  30 * time.Second,
		EnabledProfiles: []profiler.ProfileType{
			profiler.CPUProfile,
			profiler.HeapProfile,
			profiler.GoroutineProfile,
		},
	}
	
	continuousProfiler, err := profiler.NewContinuousProfiler(profilerCfg)
	if err != nil {
		log.Fatalf("Failed to create profiler: %v", err)
	}
	
	continuousProfiler.Start(ctx)
	defer continuousProfiler.Stop()
	
	// ... rest of main ...
}
```

Rebuild and re-run the server with the continuous profiling feature:

```bash
# rebuild
go build -o server ./cmd/server

# re-run the server
./server
```

Next, let's create a profile viewer tool `cmd/profview/main.go`:

```go
package main

import (
   "flag"
   "fmt"
   "log"
   "os/exec"
   "path/filepath"
   "strings"
)

var (
   profileDir  = flag.String("dir", "profiles", "Profile directory")
   profileType = flag.String("type", "cpu", "Profile type to view")
   compare     = flag.Bool("compare", false, "Compare two profiles")
)

func main() {
   flag.Parse()

   profiles, err := findProfiles(*profileDir, *profileType)
   if err != nil {
      log.Fatal(err)
   }

   if len(profiles) == 0 {
      log.Fatal("No profiles found")
   }

   switch {
   case *compare && len(profiles) >= 2:
      // Check if this profile type supports comparison
      if *profileType == "trace" {
         log.Fatal("Trace profiles cannot be compared. Use individual trace viewing instead.")
      }
      // Compare the two most recent profiles
      compareProfiles(profiles[len(profiles)-2], profiles[len(profiles)-1])
   default:
      // View the most recent profile
      viewProfile(profiles[len(profiles)-1])
   }
}

func findProfiles(dir, profileType string) ([]string, error) {
   pattern := filepath.Join(dir, fmt.Sprintf("%s_*.prof", profileType))
   return filepath.Glob(pattern)
}

func viewProfile(profile string) {
   fmt.Printf("Viewing profile: %s\n", profile)

   var cmd *exec.Cmd
   var url = "http://localhost:8084"

   // Different tools for different profile types
   switch *profileType {
   case "trace":
      // Trace files use `go tool trace`
      cmd = exec.Command("go", "tool", "trace", "-http=:8084", profile)
   default:
      // All other profiles use `go tool pprof`
      cmd = exec.Command("go", "tool", "pprof", "-http=:8084", profile)
   }

   if err := cmd.Start(); err != nil {
      log.Fatalf("Failed to start viewer: %v", err)
   }

   fmt.Printf("Profile viewer started at %s\n", url)
   if *profileType == "trace" {
      fmt.Println("Note: Trace viewer provides different views than pprof")
   }

   if err := cmd.Wait(); err != nil {
      log.Printf("Viewer process ended: %v", err)
   }
}

func compareProfiles(base, profile string) {
   fmt.Printf("Comparing profiles:\n  Base: %s\n  Current: %s\n", base, profile)

   // Check if profiles are meaningful to compare
   if strings.Contains(base, "goroutine") {
      fmt.Println("Note: Goroutine profile comparison shows changes in goroutine counts and states.")
      fmt.Println("Empty results may indicate similar goroutine patterns between captures.")
   }

   cmd := exec.Command("go", "tool", "pprof", "-http=:8085",
      fmt.Sprintf("-base=%s", base), profile)
   if err := cmd.Start(); err != nil {
      log.Fatalf("Failed to start comparison: %v", err)
   }

   fmt.Println("Profile comparison started at http://localhost:8085")
   if err := cmd.Wait(); err != nil {
      log.Printf("Comparison viewer process ended: %v", err)
   }
}
```

`Using the profile viewer tool`

Now that we have continuous profiling running and collecting profiles automatically, we need a way to analyze these profiles efficiently.
The profile viewer tool we created provides a convenient interface for examining both individual profiles and comparing profiles over time to detect performance changes.

Before we dive into using the tool, it's important to understand what it does.
The profile viewer serves as a bridge between the raw profile files collected by our continuous profiler and Go's powerful pprof visualization tools.
Instead of manually running pprof commands for each profile, our viewer automates the process and provides defaults for common analysis workflows.

`The tool operates in two primary modes`

1. **Single Profile Viewing**: Opens the most recent profile of a specified type in an interactive web interface where you can explore the performance data in detail.
2. **Profile Comparison**: Takes two profiles (typically from different time periods) and shows the differences between them, making it easy to spot performance regressions or improvements.

First, let's build the profile viewer tool and ensure we have some profiles to work with:

```bash
# Build the profile viewer tool
go build -o profview ./cmd/profview

# Verify our continuous profiler has created some profiles
ls -la profiles/

# You should see files like:
# cpu_20250524-231414.prof
# heap_20250524-231414.prof
# goroutine_20250524-231414.prof
```

If you don't see any profile files, make sure your server with continuous profiling is running and has been active for at least one collection interval (5 minutes in our configuration).

To generate some load activity for further profiling commands, we can use:

```bash
# Generate load
curl http://localhost:8080/load
```

When we use the tool, it automatically opens Go's pprof web interface at `http://localhost:8084`. This interface provides several powerful views of your profile data:

- **Top View**: Shows functions consuming the most CPU time, ranked by percentage. This is your first stop for identifying performance bottlenecks.
- **Graph View**: Displays a visual call graph where node size represents time spent and arrows show calling relationships.
  This helps you understand how different parts of your code interact and where time flows.
- **Flame Graph**: Presents a horizontal view where width represents time spent and height shows call stack depth.
  This is particularly useful for identifying deep call chains or unexpected function calls.
- **Source View**: Shows your actual source code annotated with timing information, making it easy to see which specific lines are consuming CPU time.

`Analyzing different profile types`

Each profile type reveals different aspects of your system's behavior. Let's explore how to use each one effectively:

Memory profile analysis:

```bash
# View heap memory usage
./profview -type=heap -dir=profiles
```

When examining heap profiles, focus on:
- Functions that allocate the most total memory (indicating potential optimization opportunities)
- Allocation patterns that seem excessive for the work being done
- Memory usage that doesn't align with your expectations based on the code

Goroutine profile analysis:

```bash
# View current goroutine status
./profview -type=goroutine -dir=profiles
```

Goroutine profiles help you understand:
- How many goroutines your application creates during normal operation
- Where goroutines spend their time (running, blocking, or waiting)
- Whether you have goroutine leaks (goroutines that never terminate)

Block profile analysis:

```bash
# View blocking behavior
./profview -type=block -dir=profiles
```

Block profiles reveal synchronization issues:
- Where goroutines spend time waiting for channels, mutexes, or other synchronization primitives
- Lock contention that might be limiting your application's concurrency
- Opportunities to redesign concurrent algorithms for better performance

Comparing profiles over time:

One of the features of our profile viewer is the ability to compare profiles from different periods.

```bash
# Compare the two most recent CPU profiles
./profview -type=heap -dir=profiles -compare=true
```

`Interpreting profile viewer results`

Understanding what you see in the pprof web interface is crucial for effective performance analysis. Here's how to interpret the most important visualizations:

- **Reading the top function list**: Functions are typically shown with both absolute values (time or memory) and percentages.
  Focus on functions with high percentages first, as these represent the biggest opportunities for optimization.
  However, don't ignore functions with high absolute values but low percentages if they seem unexpected for your application's workload.
- **Understanding call graphs**: In the graph view, follow the arrows to understand how execution flows through your code.
  Large nodes connected to many other large nodes often indicate central functions that coordinate significant work.
  These are prime candidates for optimization because improvements here can have wide-reaching effects.
- **Analyzing flame graphs**: Look for wide sections in the flame graph, which indicate functions that consume significant time.
  Pay particular attention to unexpected wide sections or functions that appear wider than you would expect based on your understanding of the code.

#### Step 6: Trace Analysis - Understanding Concurrency

Execution traces are like having a microscope for your Go program's runtime behavior.

While CPU and memory profiles tell you what is consuming resources, traces show you when and how things happen over time.

Traces reveal the intricate dance of goroutines, showing exactly when they start, when they block waiting for resources, when the garbage collector runs, and how the Go scheduler makes decisions.

This level of detail makes traces incredibly powerful for understanding concurrency issues, but it also makes them the most complex profiling tool to master.

In this step, we'll build our trace analysis skills progressively, starting with simple examples and working up to more complex analysis techniques that will help us diagnose performance problems.

Understanding what traces capture:

Before we dive into collecting traces, let's understand what information they contain. Go's execution tracer records detailed events during your program's execution:

1. **Goroutine Events**: When goroutines are created, when they start running, when they block, and when they complete.
   This helps you understand your program's concurrency patterns and identify goroutines that aren't behaving as expected.
2. **Scheduler Events**: How the Go scheduler distributes goroutines across available CPU cores, including scheduling delays and context switches.
   This reveals whether your concurrent design is actually achieving parallelism.
3. **Synchronization Events**: When goroutines interact through channels, mutexes, or other synchronization primitives.
   These events help you spot contention and coordination bottlenecks.
4. **Garbage Collection Events**: When the garbage collector runs, how long it takes, and how it impacts other goroutines.
   This helps you understand the relationship between memory allocation patterns and performance.
5. **System Call Events**: When your program interacts with the operating system for I/O operations or other system services.
   This helps distinguish between CPU-bound and I/O-bound performance issues.

`Our first trace: a simple concurrent example`

Let's start with a straightforward example that demonstrates basic trace collection and interpretation.

This will help us become familiar with the trace viewer interface before we tackle more complex scenarios.

Create `cmd/simple-trace/main.go`:

```go
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
```

Build and run this example:

```bash
# Build and run the simple trace example
go build -o simple-trace ./cmd/simple-trace
./simple-trace

# View the trace - this opens the web interface
go tool trace simple_trace.out
```

`Navigating the trace viewer interface`

When you run `go tool trace simple_trace.out`, you'll see a web page open in your browser, but it won't immediately show the timeline visualization.
Instead, you'll see what looks like a table of contents or index page with several links.

This initial page is the trace viewer's main menu. Think of it like opening complex analysis software where you first choose which type of analysis you want to perform.
The trace file contains enormous amounts of data, and different views help you examine different aspects of your program's performance.

Here's what you should see on this index page and what each option means:

- **View trace by proc** or **View trace by thread**: This is the main timeline visualization we want to explore first.
  Click this link to see the horizontal timeline with colored bars representing goroutine execution over time.

- **Goroutine analysis**: This provides a summary view of all goroutines, grouped by their main function. It's useful for getting an overview but isn't the visual timeline we're looking for initially.

- **Various "blocking profile" links**: These show specialized views focusing on different types of performance bottlenecks, similar to the pprof profiles we've worked with in earlier steps.

- **User-defined tasks and regions**: If your code includes custom trace annotations (which we'll add later), they appear here.

`Diagnosing concurrency issues with traces`

Let's use traces to diagnose specific concurrency problems in our document processing system. Remember the unbuffered channel issue in our `ConcurrentProcessor`?

Traces can show us exactly how this affects our system's performance.

Create `cmd/trace-diagnosis/main.go`:

```go
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
```

Build and run this diagnosis tool:

```bash
go build -o trace-diagnosis ./cmd/trace-diagnosis
./trace-diagnosis
go tool trace concurrency_diagnosis.out
```

When you run the diagnosis tool, you'll see output like:

```
2025/05/26 11:24:08 Preparing trace for viewer...
2025/05/26 11:24:08 Splitting trace for viewer...
2025/05/26 11:24:08 Opening browser. Trace viewer is listening on http://127.0.0.1:38365
```

The trace viewer will open in your browser, showing the familiar index page with multiple analysis options.

`Goroutine analysis`

The goroutine analysis view tells you exactly how each goroutine spent its time during execution.
This is often the most revealing view for diagnosing concurrency issues because it quantifies the impact of blocking behavior on individual goroutines.

From the trace viewer index page, click on `"Goroutine analysis"`. This will show you a table listing all the different types of goroutines that existed during our trace, grouped by their main function (the function that created them).

You should see entries like:
- `main.demonstrateChannelBlocking.func1` - our problematic unbuffered channel goroutines
- `main.demonstrateOptimizedProcessing.func1` - our optimized buffered channel goroutines
- Various runtime and system goroutines

Analyzing the problematic goroutines.

Click on `main.demonstrateChannelBlocking.func1` to examine the goroutines that were trying to submit documents to the unbuffered channel.

You'll see a summary at the top showing:

```
Goroutine start location: main.demonstrateChannelBlocking.func1
Count: 20
Execution time: 8.87% of total program time
```

This tells you that you created 20 of these goroutines (the 20 documents you tried to submit), and together they consumed 8.87% of your program's total CPU time.

Below the summary, you'll see a detailed breakdown table with columns for each goroutine. Understanding these columns is crucial for diagnosing performance issues:

- **Execution time**: Actual CPU time spent running your Go code. This represents productive work.
- **Block time (chan send)**: Time spent blocked trying to send on a channel. This is wasted time waiting for a receiver.
- **Block time (sync)**: Time blocked on mutexes or other synchronization primitives.
- **Sched wait time**: Time spent ready to run but waiting for a CPU core to become available.
- **Syscall execution time**: Time spent executing system calls.
- **Unknown time**: Time the tracer couldn't classify.

What the numbers reveal.

Looking at a typical goroutine from our unbuffered channel example, we might see something like:

```
Goroutine 14:
Total: 458.000 ms
Execution time: 55 µs
Block time (chan send): 457.018 ms
Sched wait time: 653 µs
Syscall execution time: 44.9 µs
```

This breakdown tells a dramatic story. Out of 458 milliseconds total lifetime:
- Only 55 microseconds were spent actually executing code (0.01% productive time)
- 457 milliseconds were spent blocked trying to send on the channel (99.8% wasted time)
- Negligible time was spent waiting for CPU scheduling or in system calls

This pattern reveals that the unbuffered channel created a severe bottleneck. Our goroutines spent virtually their entire existence parked, waiting for workers to receive from the channel.

Comparing with the optimized version.

Now click on `main.demonstrateOptimizedProcessing.func1` to see how the buffered channel version performed.

You should observe a dramatically different pattern:

```
Goroutine 25 (example):
Total: 89.000 ms
Execution time: 85.2 ms
Block time (chan send): 0 ms
Sched wait time: 3.1 ms
Syscall execution time: 0.2 ms
```

Here, the story is completely different:
- 85.2 milliseconds spent executing code (95.7% productive time)
- 0 milliseconds blocked on channel sends
- Only 3.1 milliseconds waiting for CPU scheduling

This comparison provides quantitative proof that the buffered channel eliminated the blocking bottleneck.
The goroutines went from spending 99.8% of their time blocked to spending 95.7% of their time doing productive work.

Using additional analysis tools.

The goroutine analysis page also provides links to specialized profile views:
- **Sync block profile**: Click the graph link next to this to see a visual breakdown of where synchronization blocking occurred in your code
- **Scheduler wait profile**: Shows where goroutines spent time waiting for CPU cores
- **Network wait profile**: Reveals network I/O related blocking (not relevant to this example)

These specialized views use the same visualization tools as the pprof profiles we worked with in earlier steps, making them familiar and easy to interpret.

`Practical troubleshooting patterns`

Based on what you see in goroutine analysis, here are common performance problems and their signatures:

- **Channel Contention Problem**: Large `"Block time (chan send)"` or `"Block time (chan recv)"` values indicate that your channels are creating bottlenecks.
  Solutions include buffering channels appropriately or redesigning your communication patterns.
- **Lock Contention Problem**: Significant `"Block time (sync)"` values suggest that multiple goroutines are competing for the same mutexes.
  Consider sharding your locks, using atomic operations, or redesigning your data structures to reduce contention.
- **Over-Scheduling Problem**: High `"Sched wait time"` values indicate that you have too many goroutines competing for available CPU cores.
  This suggests you should limit the number of concurrent goroutines or increase GOMAXPROCS if you have more CPU cores available.
- **I/O Bottleneck Problem**: Dominant `"Syscall execution time"` or `"Block time (syscall)"` suggests that your performance is limited by I/O operations rather than CPU work or coordination overhead.

Based on your trace analysis, here are the most common performance patterns you'll encounter and approaches for addressing them:

**1. Channel Saturation Pattern**: You observe high `"Block time (chan send)"` values. This indicates that your channels have become bottlenecks.

Solutions:
- _Buffering_: Add appropriate buffer sizes to your channels. The buffer size should match the expected burst capacity of your senders relative to the processing rate of your receivers.
- _Backpressure_: Implement non-blocking sends with timeout or dropping mechanisms when channels are full, preventing senders from accumulating.
- _Channel Sharding_: Distribute work across multiple channels to increase total throughput capacity.

**2. Lock Contention Pattern**: Significant `"Block time (sync)"` values appear in your goroutine analysis, and you see frequent blocking events clustered around mutex operations in the timeline.

Solutions:
- _Lock Granularity_: Replace coarse-grained locks protecting large data structures with fine-grained locks protecting smaller sections.
- _Atomic Operations_: For simple operations like counters or flags, replace mutex-protected operations with atomic operations.
- _Read-Write Locks_: When you have many readers and few writers, `RWMutex` can significantly reduce contention.
- _Lock-Free Data Structures_: For high-contention scenarios, consider redesigning your data structures to avoid locks entirely.

**3. Over-Goroutine Pattern**: High `"Sched wait time"` values and timeline views showing many short-lived goroutines competing for processor time indicate that you're creating more goroutines than your system can efficiently schedule.

Solutions:
- _Worker Pools_: Replace unlimited goroutine creation with a fixed pool of worker goroutines that process work from channels.
- _Rate Limiting_: Control the rate at which you create new goroutines based on system capacity.
- _Goroutine Lifecycle Management_: Ensure goroutines complete their work promptly and don't accumulate in the system.

**4. Memory Pressure Pattern**: Frequent garbage collection events in the timeline view, combined with goroutines showing unexpected blocking behavior during GC pauses.

Solutions:
- _Allocation Reduction_: Minimize memory allocations in hot paths by reusing objects and using object pools.
- _Batch Processing_: Process work in larger batches to reduce the allocation rate and give the garbage collector more time between intense periods.
- _Memory-Efficient Data Structures_: Choose data structures that minimize memory overhead and fragmentation.

`Adding custom trace events for deeper insight`

The built-in trace events are powerful, but sometimes you need to add your own markers to understand specific aspects of your application's behavior.
Go's trace package allows you to add custom events that appear in the timeline, helping you understand the flow of work through your system.

Now let's create a comprehensive trace that demonstrates multiple performance scenarios with custom events:

Create `cmd/comprehensive-trace/main.go`:

```go
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
```

Build and run the comprehensive trace tool:

```bash
go build -o comprehensive-trace ./cmd/comprehensive-trace
./comprehensive-trace
go tool trace comprehensive_performance.out
```

This trace demonstrates four distinct performance patterns that you'll commonly encounter in applications.

**1. Interpreting the channel contention section**:

Look for goroutines that start and immediately enter blocked states.

You should see the producer goroutines forming a "waiting line" (`Goroutines` -> `main.channelContentionScenario.func2`, `Block time(chan send)` column) because they can't send to the unbuffered channel until the slow consumer is ready.
This creates a serialization pattern where potential parallelism is lost to coordination overhead.

**2. Understanding the lock contention section**:

In this section, you'll see multiple goroutines competing for the same mutex (`Goroutines` -> `main.lockContentionScenario.func1`, `Block time (sync)` column).
Look for periods where several goroutines are in runnable state, but only one is actually running (holding the lock). The trace will show clear serialization of work that could potentially be parallel.

**3. Observing memory pressure effects**:

Watch the GC timeline during this section. You should see more frequent garbage collection events, and you might notice that other goroutines pause during GC runs.
This demonstrates how memory allocation patterns can impact overall system performance.

**4. Recognizing optimal concurrency patterns**:

This section should show smooth, parallel execution with minimal blocking.
Worker goroutines should show consistent utilization without excessive coordination overhead. This represents the performance characteristics you want to achieve in your concurrent designs.

#### Conclusion

Throughout this deep dive into Go's profiling capabilities, we've explored a systematic approach to identifying and fixing performance bottlenecks in Go applications.

We started with the fundamentals of profiling and progressively built up to advanced techniques for diagnosing complex concurrency issues.

The journey took us through several key aspects of performance optimization.

First, we examined CPU profiling to identify computational hot spots, learning how to locate inefficient algorithms and excessive function calls.
Our optimizations demonstrated how seemingly small changes - like pre-compiling regular expressions, reducing string allocations, and implementing efficient data structures - can lead to significant performance improvements.

Next, we explored memory profiling to uncover allocation patterns and potential memory leaks. By implementing bounded caches and object pools, we saw how thoughtful memory management can prevent resource exhaustion and reduce garbage collection pressure.

We then tackled concurrency issues using goroutine and block profiling, identifying synchronization bottlenecks and lock contention. The transition from unbuffered to buffered channels and from mutex-protected counters to atomic operations showed how proper concurrency design dramatically improves throughput and responsiveness.

Our exploration of execution traces provided a microscopic view of our application's runtime behavior, revealing intricate interactions between goroutines, the scheduler, and the garbage collector.
The custom trace events demonstrated how to gain deeper insights into application-specific behavior patterns.

Finally, we established continuous profiling infrastructure to catch performance regressions early, making performance optimization an ongoing part of the development lifecycle rather than a one-time effort.

The most important takeaway is that performance optimization is not about blindly applying "best practices" but about measurement-driven decisions.

The profiling tools we've explored allow us to:

1. Quantify performance issues rather than relying on intuition
2. Focus efforts on the bottlenecks that matter most
3. Verify that our optimizations actually improve performance
4. Build a feedback loop that prevents performance regressions

Every production Go application will eventually face performance challenges as it scales and evolves. The difference between applications that maintain high performance and those that degrade lies in the team's ability to diagnose and address issues systematically.

By integrating profiling into your development workflow, you gain a superpower: the ability to peer inside your running application and understand exactly what's happening at the deepest levels.
This insight transforms performance optimization from guesswork into precision engineering.

Remember that optimization is a journey, not a destination. As your application evolves, new performance challenges will emerge.
The profiling techniques we've explored provide a reliable compass to navigate these challenges and maintain the high-performance systems your users expect.