package events

import (
	"encoding/json"
	"fmt"
	"sync/atomic"
	"time"

	"saga-choreography/pkg/models"
)

// EventType identifies the type of event being published.
type EventType string

// Domain events for the order saga.
//
// The naming convention follows: {Entity}{Action}
// Events use past tense because they represent something that has already happened.
// This is a deliberate design choice: events are facts about the world, not commands.
const (
	// Order events
	OrderCreated   EventType = "order.created"
	OrderCompleted EventType = "order.completed"
	OrderCancelled EventType = "order.cancelled"

	// Payment events
	PaymentCompleted EventType = "payment.completed"
	PaymentFailed    EventType = "payment.failed"
	PaymentRefunded  EventType = "payment.refunded"

	// Inventory events
	InventoryReserved      EventType = "inventory.reserved"
	InventoryReserveFailed EventType = "inventory.reserve_failed"
	InventoryReleased      EventType = "inventory.released"

	// Shipping events
	ShipmentScheduled      EventType = "shipment.scheduled"
	ShipmentScheduleFailed EventType = "shipment.schedule_failed"
	ShipmentCancelled      EventType = "shipment.cancelled"
)

// idCounter provides monotonically increasing IDs that are safe for concurrent use.
// Using time-based IDs (like time.Now().Format(...)) leads to collisions when multiple
// events are created within the same nanosecond, which is common in event-driven systems
// where a single incoming event can trigger several outgoing events in rapid succession.
var idCounter atomic.Uint64

// GenerateID creates a unique identifier by combining a monotonic counter with a timestamp.
// The counter guarantees uniqueness even under high concurrency, while the timestamp
// provides human-readable ordering for debugging. In production, use a proper UUID library.
func GenerateID(prefix string) string {
	id := idCounter.Add(1)
	return fmt.Sprintf("%s-%d-%04d", prefix, time.Now().UnixMilli(), id)
}

// Event represents a domain event in our saga.
// Every event carries two tracing IDs that together let you reconstruct the full
// causal history of any saga instance.
type Event struct {
	ID            string          `json:"id"`
	Type          EventType       `json:"type"`
	AggregateID   string          `json:"aggregate_id"`   // The ID of the entity this event relates to
	AggregateType string          `json:"aggregate_type"` // The type of entity (order, payment, etc.)
	Timestamp     time.Time       `json:"timestamp"`
	Data          json.RawMessage `json:"data"`

	// CorrelationID groups all events in a single saga instance.
	// Typically set to the order ID so that every event produced during
	// order processing, across all services, shares the same correlation ID.
	CorrelationID string `json:"correlation_id"`

	// CausationID records the ID of the event that directly caused this event.
	// While CorrelationID gives you the "which saga" answer, CausationID gives
	// you the "why did this happen" answer, forming a chain of cause and effect.
	CausationID string `json:"causation_id"`
}

// NewEvent creates a new event with the given type and data.
func NewEvent(eventType EventType, aggregateID, aggregateType string, data any) (*Event, error) {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal event data: %w", err)
	}

	return &Event{
		ID:            GenerateID("evt"),
		Type:          eventType,
		AggregateID:   aggregateID,
		AggregateType: aggregateType,
		Timestamp:     time.Now(),
		Data:          dataBytes,
	}, nil
}

// WithCorrelation sets the correlation ID for event tracing.
func (e *Event) WithCorrelation(correlationID string) *Event {
	e.CorrelationID = correlationID
	return e
}

// WithCausation sets the causation ID to track event chains.
func (e *Event) WithCausation(causationID string) *Event {
	e.CausationID = causationID
	return e
}

// ParseData unmarshals the event data into the provided struct.
func (e *Event) ParseData(v any) error {
	return json.Unmarshal(e.Data, v)
}

// --- Event payload structures ---
// These define the data carried by each event type.
// A critical design decision in choreography is what data to include in each event.
// Including too little forces services to query each other (introducing coupling).
// Including too much creates large events and potential data consistency issues.
// The guideline: include everything a downstream consumer needs to act autonomously.

// OrderCreatedData contains the data for an order.created event.
// We include the full order with all items because multiple downstream services
// (Payment and Inventory) need this information to do their work.
type OrderCreatedData struct {
	Order models.Order `json:"order"`
}

// PaymentCompletedData contains the data for a payment.completed event.
type PaymentCompletedData struct {
	PaymentID  string  `json:"payment_id"`
	OrderID    string  `json:"order_id"`
	CustomerID string  `json:"customer_id"`
	Amount     float64 `json:"amount"`
}

// PaymentFailedData contains the data for a payment.failed event.
type PaymentFailedData struct {
	OrderID    string `json:"order_id"`
	CustomerID string `json:"customer_id"`
	Reason     string `json:"reason"`
}

// PaymentRefundedData contains the data for a payment.refunded event.
type PaymentRefundedData struct {
	PaymentID  string  `json:"payment_id"`
	OrderID    string  `json:"order_id"`
	CustomerID string  `json:"customer_id"`
	Amount     float64 `json:"amount"`
	Reason     string  `json:"reason"`
}

// InventoryReservedData contains the data for an inventory.reserved event.
type InventoryReservedData struct {
	ReservationID string                            `json:"reservation_id"`
	OrderID       string                            `json:"order_id"`
	Items         []models.InventoryReservationItem `json:"items"`
}

// InventoryReserveFailedData contains the data for an inventory.reserve_failed event.
type InventoryReserveFailedData struct {
	OrderID string `json:"order_id"`
	Reason  string `json:"reason"`
}

// InventoryReleasedData contains the data for an inventory.released event.
type InventoryReleasedData struct {
	ReservationID string `json:"reservation_id"`
	OrderID       string `json:"order_id"`
	Reason        string `json:"reason"`
}

// ShipmentScheduledData contains the data for a shipment.scheduled event.
type ShipmentScheduledData struct {
	ShipmentID     string `json:"shipment_id"`
	OrderID        string `json:"order_id"`
	TrackingNumber string `json:"tracking_number"`
}

// ShipmentScheduleFailedData contains the data for a shipment.schedule_failed event.
type ShipmentScheduleFailedData struct {
	OrderID string `json:"order_id"`
	Reason  string `json:"reason"`
}

// ShipmentCancelledData contains the data for a shipment.cancelled event.
type ShipmentCancelledData struct {
	ShipmentID string `json:"shipment_id"`
	OrderID    string `json:"order_id"`
	Reason     string `json:"reason"`
}

// OrderCompletedData contains the data for an order.completed event.
type OrderCompletedData struct {
	OrderID        string `json:"order_id"`
	PaymentID      string `json:"payment_id"`
	ReservationID  string `json:"reservation_id"`
	ShipmentID     string `json:"shipment_id"`
	TrackingNumber string `json:"tracking_number"`
}

// OrderCancelledData contains the data for an order.cancelled event.
type OrderCancelledData struct {
	OrderID string `json:"order_id"`
	Reason  string `json:"reason"`
}
