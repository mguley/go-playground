package monitor

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// SagaState represents the current state of a saga instance.
type SagaState struct {
	CorrelationID string
	OrderID       string
	Status        string
	StartTime     time.Time
	LastEventTime time.Time
	Events        []EventRecord
	FinalStatus   models.OrderStatus
}

// EventRecord captures information about an event in the saga.
type EventRecord struct {
	EventID   string
	EventType events.EventType
	Timestamp time.Time
	Source    string
}

// SagaMetrics contains aggregate metrics about saga execution.
type SagaMetrics struct {
	TotalSagas      int
	CompletedSagas  int
	CancelledSagas  int
	InProgressSagas int
	AverageDuration time.Duration
}

// SagaMonitor tracks the state of all saga instances.
type SagaMonitor struct {
	mu       sync.RWMutex
	sagas    map[string]*SagaState
	eventBus *events.EventBus
	logger   *slog.Logger
}

// NewSagaMonitor creates a new saga monitor.
func NewSagaMonitor(eventBus *events.EventBus, logger *slog.Logger) *SagaMonitor {
	m := &SagaMonitor{
		sagas:    make(map[string]*SagaState),
		eventBus: eventBus,
		logger:   logger,
	}

	m.subscribeToAllEvents()
	return m
}

// subscribeToAllEvents subscribes to all saga-related events.
// The monitor is a passive observer: it records everything but never
// produces events or influences the saga flow. This is important because
// a monitor that participated in the saga could become a single point of
// failure, contradicting one of the primary benefits of choreography.
func (m *SagaMonitor) subscribeToAllEvents() {
	allEventTypes := []events.EventType{
		events.OrderCreated,
		events.OrderCompleted,
		events.OrderCancelled,
		events.PaymentCompleted,
		events.PaymentFailed,
		events.PaymentRefunded,
		events.InventoryReserved,
		events.InventoryReserveFailed,
		events.InventoryReleased,
		events.ShipmentScheduled,
		events.ShipmentScheduleFailed,
		events.ShipmentCancelled,
	}

	m.eventBus.Subscribe("saga-monitor", allEventTypes, m.handleEvent)
}

// handleEvent records events for monitoring.
func (m *SagaMonitor) handleEvent(ctx context.Context, event *events.Event) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	correlationID := event.CorrelationID
	if correlationID == "" {
		return nil
	}

	saga, exists := m.sagas[correlationID]
	if !exists {
		saga = &SagaState{
			CorrelationID: correlationID,
			OrderID:       event.AggregateID,
			Status:        "IN_PROGRESS",
			StartTime:     event.Timestamp,
			Events:        make([]EventRecord, 0),
		}
		m.sagas[correlationID] = saga
	}

	// Record the event
	saga.Events = append(saga.Events, EventRecord{
		EventID:   event.ID,
		EventType: event.Type,
		Timestamp: event.Timestamp,
		Source:    event.AggregateType,
	})
	saga.LastEventTime = event.Timestamp

	// Update saga status based on terminal events
	switch event.Type {
	case events.OrderCompleted:
		saga.Status = "COMPLETED"
		saga.FinalStatus = models.OrderStatusCompleted
	case events.OrderCancelled:
		saga.Status = "CANCELLED"
		saga.FinalStatus = models.OrderStatusCancelled
	}

	return nil
}

// GetSagaState returns the state of a specific saga.
func (m *SagaMonitor) GetSagaState(correlationID string) (*SagaState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	saga, exists := m.sagas[correlationID]
	if !exists {
		return nil, fmt.Errorf("saga %s not found", correlationID)
	}

	// Return a copy
	sagaCopy := new(*saga)
	sagaCopy.Events = make([]EventRecord, len(saga.Events))
	copy(sagaCopy.Events, saga.Events)

	return sagaCopy, nil
}

// GetAllSagas returns all tracked sagas.
func (m *SagaMonitor) GetAllSagas() []*SagaState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sagas := make([]*SagaState, 0, len(m.sagas))
	for _, saga := range m.sagas {
		sagaCopy := new(*saga)
		sagaCopy.Events = make([]EventRecord, len(saga.Events))
		copy(sagaCopy.Events, saga.Events)
		sagas = append(sagas, sagaCopy)
	}

	return sagas
}

// GetMetrics returns aggregate metrics about saga execution.
func (m *SagaMonitor) GetMetrics() SagaMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	metrics := SagaMetrics{
		TotalSagas: len(m.sagas),
	}

	var totalDuration time.Duration
	completedCount := 0

	for _, saga := range m.sagas {
		switch saga.Status {
		case "COMPLETED":
			metrics.CompletedSagas++
		case "CANCELLED":
			metrics.CancelledSagas++
		case "IN_PROGRESS":
			metrics.InProgressSagas++
		}

		if saga.Status != "IN_PROGRESS" {
			duration := saga.LastEventTime.Sub(saga.StartTime)
			totalDuration += duration
			completedCount++
		}
	}

	if completedCount > 0 {
		metrics.AverageDuration = totalDuration / time.Duration(completedCount)
	}

	return metrics
}

// GetStuckSagas returns sagas that have been in progress longer than the given threshold.
// In production, this would feed into an alerting system. Stuck sagas indicate either
// a lost event (the message broker dropped it), a crashed service, or a defect in the
// event flow that leaves the saga in a non-terminal state.
//
// This method is the primary defense against the "lost wake-up" or "phantom event"
// problem in choreography. Consider what happens if payment.completed is published
// but the event broker goes down before the Inventory service receives it: the saga
// will remain IN_PROGRESS indefinitely because no downstream service ever acts.
// A production system would pair this detection with a background "sweeper" job
// that periodically queries stuck sagas, inspects their last known event, and
// republishes that event (or a synthetic recovery event) to unstick the flow.
// The combination of stuck-saga detection and event replay forms a self-healing
// mechanism that makes choreography viable for production workloads.
func (m *SagaMonitor) GetStuckSagas(threshold time.Duration) []*SagaState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var stuck []*SagaState
	now := time.Now()
	for _, saga := range m.sagas {
		if saga.Status == "IN_PROGRESS" && now.Sub(saga.StartTime) > threshold {
			stuck = append(stuck, new(*saga))
		}
	}
	return stuck
}

// PrintSagaTimeline prints a visual timeline of a saga.
func (m *SagaMonitor) PrintSagaTimeline(correlationID string) {
	saga, err := m.GetSagaState(correlationID)
	if err != nil {
		m.logger.Error("Failed to get saga state",
			"correlation_id", correlationID,
			"error", err,
		)
		return
	}

	fmt.Printf("\n=== Saga Timeline: %s ===\n", correlationID)
	fmt.Printf("Status: %s\n", saga.Status)
	fmt.Printf("Duration: %v\n\n", saga.LastEventTime.Sub(saga.StartTime))

	for i, event := range saga.Events {
		offset := event.Timestamp.Sub(saga.StartTime)
		fmt.Printf("[+%8v] %s\n", offset.Round(time.Millisecond), event.EventType)

		if i < len(saga.Events)-1 {
			fmt.Println("     |")
			fmt.Println("     v")
		}
	}

	fmt.Println()
}
