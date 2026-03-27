package es

import (
	"time"
)

// Event represents a single domain event in the system. Events are
// immutable records of something that happened. They are the source
// of truth in an event-sourced system - the current state of any
// entity is derived by replaying its events from the beginning.
//
// The Version field is crucial for optimistic concurrency control:
// when saving new events, the store checks that the expected version
// matches the current version of the aggregate, preventing lost-update
// problems from concurrent command processing.
type Event struct {
	// ID uniquely identifies this event. Assigned by the event store
	// when the event is persisted.
	ID string

	// AggregateID identifies which aggregate (entity) this event
	// belongs to. All events for a single bank account share the
	// same AggregateID.
	AggregateID string

	// AggregateType is the type name of the aggregate (e.g., "Account").
	// This allows a single event store to host multiple aggregate types.
	AggregateType string

	// EventType is the name of the event (e.g., "AccountOpened",
	// "MoneyDeposited"). Used for deserialization and routing.
	EventType string

	// Version is the sequence number of this event within its aggregate
	// stream. The first event for an aggregate has version 1, the second
	// has version 2, and so on. Versions must be contiguous - gaps
	// indicate data corruption or a defect.
	Version int

	// Timestamp records when the event occurred.
	Timestamp time.Time

	// Data holds the event-specific payload. The concrete type depends
	// on EventType - for example, a "MoneyDeposited" event's Data field
	// holds a MoneyDepositedEvent struct.
	Data any
}
