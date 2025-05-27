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
