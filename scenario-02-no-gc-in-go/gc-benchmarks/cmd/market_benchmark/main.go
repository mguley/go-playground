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
