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
