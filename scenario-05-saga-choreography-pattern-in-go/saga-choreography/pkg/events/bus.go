package events

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// EventHandler is a function that processes events.
type EventHandler func(ctx context.Context, event *Event) error

// Subscription represents a subscription to specific event types.
type Subscription struct {
	ID          string
	ServiceName string
	EventTypes  []EventType
	Handler     EventHandler
}

// EventBus manages event publishing and subscription.
// This is an in-memory implementation for demonstration purposes.
// In production, replace this with Kafka, NATS, or RabbitMQ.
type EventBus struct {
	mu            sync.RWMutex
	subscriptions map[EventType][]*Subscription
	allEvents     []*Event // Store all events for debugging/replay

	// done is closed when a terminal event (OrderCompleted or OrderCancelled) is published.
	done chan struct{}

	// Configuration
	asyncDelivery bool
	logger        *slog.Logger
}

// NewEventBus creates a new event bus.
func NewEventBus(logger *slog.Logger) *EventBus {
	return &EventBus{
		subscriptions: make(map[EventType][]*Subscription),
		allEvents:     make([]*Event, 0),
		done:          make(chan struct{}, 100), // buffered so multiple sagas can signal
		asyncDelivery: true,                     // default to async for realistic behavior
		logger:        logger,
	}
}

// SetAsyncDelivery controls whether events are delivered asynchronously.
// Synchronous delivery is useful for testing because it makes event processing
// deterministic and eliminates the need for sleeps or polling.
func (eb *EventBus) SetAsyncDelivery(async bool) {
	eb.asyncDelivery = async
}

// WaitForSaga blocks until a terminal event is published or the context is cancelled.
func (eb *EventBus) WaitForSaga(ctx context.Context, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	select {
	case <-eb.done:
		// Give async handlers a moment to finish processing the terminal event.
		time.Sleep(50 * time.Millisecond)
		return nil
	case <-ctx.Done():
		return fmt.Errorf("saga did not complete within %v", timeout)
	}
}

// Subscribe registers a handler for specific event types.
func (eb *EventBus) Subscribe(serviceName string, eventTypes []EventType, handler EventHandler) *Subscription {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	sub := &Subscription{
		ID:          GenerateID("sub"),
		ServiceName: serviceName,
		EventTypes:  eventTypes,
		Handler:     handler,
	}

	for _, eventType := range eventTypes {
		eb.subscriptions[eventType] = append(eb.subscriptions[eventType], sub)
		eb.logger.Info("[EventBus] Subscribed to event",
			"service", serviceName,
			"event_type", string(eventType),
		)
	}

	return sub
}

// Publish sends an event to all interested subscribers.
func (eb *EventBus) Publish(ctx context.Context, event *Event) error {
	eb.mu.Lock()
	eb.allEvents = append(eb.allEvents, event)

	// Copy the subscriber slice so we can release the lock before delivering.
	// Without this copy, we'd hold the lock during handler execution, which would
	// deadlock if a handler tries to publish another event (which they all do).
	subscribers := make([]*Subscription, len(eb.subscriptions[event.Type]))
	copy(subscribers, eb.subscriptions[event.Type])
	eb.mu.Unlock()

	eb.logger.Info("[EventBus] Publishing event",
		"event_type", string(event.Type),
		"event_id", event.ID,
		"correlation_id", event.CorrelationID,
	)

	// Signal completion for terminal events so WaitForSaga can unblock
	if event.Type == OrderCompleted || event.Type == OrderCancelled {
		select {
		case eb.done <- struct{}{}:
		default:
		}
	}

	if len(subscribers) == 0 {
		eb.logger.Warn("[EventBus] No subscribers for event",
			"event_type", string(event.Type),
		)
		return nil
	}

	// Deliver to all subscribers
	for _, sub := range subscribers {
		if eb.asyncDelivery {
			// Async delivery: more realistic for distributed systems.
			// Each subscriber processes events independently and concurrently.
			go eb.deliverEvent(ctx, sub, event)
		} else {
			// Sync delivery: useful for testing where you need deterministic ordering.
			if err := eb.deliverEvent(ctx, sub, event); err != nil {
				return err
			}
		}
	}

	return nil
}

// deliverEvent delivers an event to a single subscriber.
func (eb *EventBus) deliverEvent(ctx context.Context, sub *Subscription, event *Event) error {
	eb.logger.Info("[EventBus] Delivering event",
		"event_type", string(event.Type),
		"target_service", sub.ServiceName,
	)

	if err := sub.Handler(ctx, event); err != nil {
		eb.logger.Error("[EventBus] Event delivery failed",
			"event_type", string(event.Type),
			"target_service", sub.ServiceName,
			"error", err,
		)
		return fmt.Errorf("handler error in %s: %w", sub.ServiceName, err)
	}

	return nil
}

// GetEvents returns all published events (for debugging).
func (eb *EventBus) GetEvents() []*Event {
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	events := make([]*Event, len(eb.allEvents))
	copy(events, eb.allEvents)
	return events
}

// GetEventsByCorrelation returns all events for a specific saga instance.
func (eb *EventBus) GetEventsByCorrelation(correlationID string) []*Event {
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	var events []*Event
	for _, event := range eb.allEvents {
		if event.CorrelationID == correlationID {
			events = append(events, event)
		}
	}
	return events
}

// Reset clears all events and creates a fresh completion channel.
// Call this between test scenarios to get a clean slate.
func (eb *EventBus) Reset() {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.allEvents = make([]*Event, 0)
	eb.done = make(chan struct{}, 100)
}
