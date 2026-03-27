package es

import (
	"fmt"
	"sync"
	"time"

	"crypto/rand"
	"encoding/hex"
)

// EventStore defines the interface for persisting and retrieving events.
// Any implementation (in-memory, PostgreSQL, EventStoreDB) must satisfy
// this contract.
type EventStore interface {
	// SaveEvents appends new events to the aggregate's stream.
	//
	// The expectedVersion parameter enables optimistic concurrency
	// control: if the current version of the aggregate in the store
	// does not match expectedVersion, the save is rejected with a
	// ConcurrencyError. This prevents two concurrent command handlers
	// from silently overwriting each other's changes.
	SaveEvents(aggregateID string, events []Event, expectedVersion int) error

	// LoadEvents retrieves all events for a given aggregate, ordered
	// by version. The returned events can be replayed to reconstruct
	// the aggregate's current state.
	LoadEvents(aggregateID string) ([]Event, error)

	// LoadEventsFrom retrieves events for a given aggregate starting
	// from a specific version (inclusive). This is used in combination
	// with snapshots: load the snapshot (which records the version it
	// was taken at), then load only events after that version.
	LoadEventsFrom(aggregateID string, fromVersion int) ([]Event, error)
}

// ConcurrencyError is returned when a save operation fails because
// another process has already written events past the expected version.
// This is not a defect - it is the optimistic concurrency control mechanism
// doing its job. The caller should reload the aggregate, re-evaluate the command, and retry.
type ConcurrencyError struct {
	AggregateID     string
	ExpectedVersion int
	ActualVersion   int
}

func (e *ConcurrencyError) Error() string {
	return fmt.Sprintf(
		"concurrency conflict on aggregate %s: expected version %d, but current version is %d",
		e.AggregateID, e.ExpectedVersion, e.ActualVersion,
	)
}

// InMemoryEventStore is a thread-safe, in-memory implementation of
// EventStore. It stores events in a map keyed by aggregate ID, with
// each value being an ordered slice of events.
//
// This implementation is suitable for testing, prototyping, and
// learning. A production system would replace it with a durable store
// backed by a database.
type InMemoryEventStore struct {
	mu     sync.RWMutex
	events map[string][]Event // aggregateID -> ordered events

	// subscribers receive every event that is successfully saved.
	// This is the mechanism by which read-model projections stay
	// in sync with the write model.
	subscribers []func(Event)
}

// NewInMemoryEventStore creates a new empty event store.
func NewInMemoryEventStore() *InMemoryEventStore {
	return &InMemoryEventStore{
		events: make(map[string][]Event),
	}
}

// Subscribe registers a callback that will be invoked for every
// event that is successfully persisted. Projections use this to
// update their read models in response to new events.
//
// In a production system, this would be replaced by a durable
// subscription mechanism (e.g., polling the event store, a message
// broker, or a change data capture stream).
func (s *InMemoryEventStore) Subscribe(handler func(Event)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.subscribers = append(s.subscribers, handler)
}

// SaveEvents persists new events with optimistic concurrency control.
// It validates that the expected version matches the current stream
// length, assigns event metadata (ID, timestamp, version), and
// notifies all subscribers.
func (s *InMemoryEventStore) SaveEvents(aggregateID string, events []Event, expectedVersion int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	existing := s.events[aggregateID]
	currentVersion := len(existing)

	// Optimistic concurrency check: the caller must know the current
	// version to append new events. If another goroutine has already
	// appended events since the caller loaded the aggregate, this
	// check fails and the caller must retry.
	if currentVersion != expectedVersion {
		return &ConcurrencyError{
			AggregateID:     aggregateID,
			ExpectedVersion: expectedVersion,
			ActualVersion:   currentVersion,
		}
	}

	// Assign metadata to each event and append to the stream.
	enriched := make([]Event, len(events))
	for i, event := range events {
		event.ID = generateEventID()
		event.Version = currentVersion + i + 1
		event.Timestamp = time.Now()
		enriched[i] = event
	}

	s.events[aggregateID] = append(existing, enriched...)

	// Notify subscribers. In production you would notify outside
	// the critical section, but for simplicity we notify while
	// holding the lock. This guarantees subscribers see events
	// in order.
	for _, event := range enriched {
		for _, sub := range s.subscribers {
			sub(event)
		}
	}

	return nil
}

// LoadEvents returns all events for an aggregate, ordered by version.
func (s *InMemoryEventStore) LoadEvents(aggregateID string) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	events := s.events[aggregateID]
	if len(events) == 0 {
		return nil, nil
	}

	// Return a copy to prevent mutation of the store's internal data.
	result := make([]Event, len(events))
	copy(result, events)
	return result, nil
}

// LoadEventsFrom returns events starting from a specific version
// (inclusive). Used for loading events after a snapshot.
func (s *InMemoryEventStore) LoadEventsFrom(aggregateID string, fromVersion int) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	allEvents := s.events[aggregateID]
	if len(allEvents) == 0 {
		return nil, nil
	}

	// Events are 1-indexed by version but stored 0-indexed in the slice.
	startIdx := fromVersion - 1
	if startIdx < 0 {
		startIdx = 0
	}
	if startIdx >= len(allEvents) {
		return nil, nil
	}

	result := make([]Event, len(allEvents)-startIdx)
	copy(result, allEvents[startIdx:])
	return result, nil
}

// generateEventID creates a random hex string for event identification.
func generateEventID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
