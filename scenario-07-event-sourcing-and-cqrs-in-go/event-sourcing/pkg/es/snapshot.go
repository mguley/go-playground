package es

import (
	"sync"
	"time"
)

// Snapshot represents a point in time capture of an aggregate's state.
// Instead of replaying all events from the beginning, the system can
// load the most recent snapshot and replay only events since that
// version.
//
// For an account with 10,000 events and a snapshot at version 9,950,
// loading the aggregate requires deserializing one snapshot and
// replaying 50 events, rather than replaying all 10,000.
type Snapshot struct {
	AggregateID   string
	AggregateType string
	Version       int // the event version this snapshot represents
	State         any // the serialized aggregate state
	Timestamp     time.Time
}

// SnapshotStore defines the interface for persisting and retrieving
// snapshots. In production, this would be backed by a database
// (often the same one as the event store, in a separate table).
type SnapshotStore interface {
	SaveSnapshot(snapshot Snapshot) error
	LoadSnapshot(aggregateID string) (*Snapshot, error)
}

// InMemorySnapshotStore is a thread-safe, in-memory snapshot store.
type InMemorySnapshotStore struct {
	mu        sync.RWMutex
	snapshots map[string]Snapshot
}

// NewInMemorySnapshotStore creates a new empty snapshot store.
func NewInMemorySnapshotStore() *InMemorySnapshotStore {
	return &InMemorySnapshotStore{
		snapshots: make(map[string]Snapshot),
	}
}

// SaveSnapshot stores (or replaces) the snapshot for an aggregate.
func (s *InMemorySnapshotStore) SaveSnapshot(snapshot Snapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	snapshot.Timestamp = time.Now()
	s.snapshots[snapshot.AggregateID] = snapshot
	return nil
}

// LoadSnapshot retrieves the most recent snapshot for an aggregate.
// Returns nil if no snapshot exists.
func (s *InMemorySnapshotStore) LoadSnapshot(aggregateID string) (*Snapshot, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	snap, exists := s.snapshots[aggregateID]
	if !exists {
		return nil, nil
	}
	return &snap, nil
}
