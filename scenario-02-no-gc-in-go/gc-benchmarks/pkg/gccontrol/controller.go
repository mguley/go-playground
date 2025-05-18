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
