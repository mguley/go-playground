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
