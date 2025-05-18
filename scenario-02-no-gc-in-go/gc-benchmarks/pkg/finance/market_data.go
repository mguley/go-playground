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
