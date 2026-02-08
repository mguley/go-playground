package middleware

import (
	"encoding/json"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// Metrics collects request metrics for monitoring.
type Metrics struct {
	RequestCount   atomic.Int64
	ErrorCount     atomic.Int64
	TotalLatencyMs atomic.Int64

	mu             sync.RWMutex
	endpointCounts map[string]int64
}

func NewMetrics() *Metrics {
	return &Metrics{
		endpointCounts: make(map[string]int64),
	}
}

// Middleware returns HTTP middleware that collects request metrics.
func (m *Metrics) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		wrapped := &responseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
		}

		next.ServeHTTP(wrapped, r)

		m.RequestCount.Add(1)
		m.TotalLatencyMs.Add(time.Since(start).Milliseconds())

		if wrapped.statusCode >= 400 {
			m.ErrorCount.Add(1)
		}

		m.mu.Lock()
		m.endpointCounts[r.URL.Path]++
		m.mu.Unlock()
	})
}

// Handler exposes collected metrics as JSON via GET /metrics.
func (m *Metrics) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.RLock()
		endpoints := make(map[string]int64)
		for k, v := range m.endpointCounts {
			endpoints[k] = v
		}
		m.mu.RUnlock()

		requestCount := m.RequestCount.Load()
		avgLatency := float64(0)
		if requestCount > 0 {
			avgLatency = float64(m.TotalLatencyMs.Load()) / float64(requestCount)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"request_count":   requestCount,
			"error_count":     m.ErrorCount.Load(),
			"avg_latency_ms":  avgLatency,
			"endpoint_counts": endpoints,
			"collection_time": time.Now().Format(time.RFC3339),
		})
	})
}
