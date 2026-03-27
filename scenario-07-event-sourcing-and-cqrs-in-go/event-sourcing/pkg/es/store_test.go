package es

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

// makeTestEvent builds a minimal Event suitable for store tests. The store
// assigns ID, Version, and Timestamp on save, so we only need the fields
// that the caller controls.
func makeTestEvent(aggregateID, eventType string, data any) Event {
	return Event{
		AggregateID:   aggregateID,
		AggregateType: "TestAggregate",
		EventType:     eventType,
		Data:          data,
	}
}

// saveN is a shorthand that saves `n` events to the given aggregate,
// starting from expectedVersion 0 (empty stream) through sequential
// single-event batches. It fails the test if any save returns an error.
func saveN(t *testing.T, store *InMemoryEventStore, aggregateID string, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		evt := makeTestEvent(aggregateID, "Deposited", map[string]int{"seq": i})
		if err := store.SaveEvents(aggregateID, []Event{evt}, i); err != nil {
			t.Fatalf("saveN: failed on event %d: %v", i, err)
		}
	}
}

// Saving a single event to an empty stream should succeed and assign
// version 1, a non-empty ID, and a non-zero timestamp.
func TestSaveEvents_SingleEvent(t *testing.T) {
	store := NewInMemoryEventStore()

	evt := makeTestEvent("agg-1", "AccountOpened", "payload")
	if err := store.SaveEvents("agg-1", []Event{evt}, 0); err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	events, _ := store.LoadEvents("agg-1")
	if len(events) != 1 {
		t.Fatalf("expected 1 stored event, got %d", len(events))
	}

	saved := events[0]
	if saved.ID == "" {
		t.Error("expected store to assign a non-empty event ID")
	}
	if saved.Version != 1 {
		t.Errorf("expected version 1, got %d", saved.Version)
	}
	if saved.Timestamp.IsZero() {
		t.Error("expected store to assign a non-zero timestamp")
	}
	if saved.AggregateID != "agg-1" {
		t.Errorf("expected aggregate ID 'agg-1', got %q", saved.AggregateID)
	}
	if saved.EventType != "AccountOpened" {
		t.Errorf("expected event type 'AccountOpened', got %q", saved.EventType)
	}
}

// Saving a batch of multiple events in one call should assign
// sequential versions starting after the current stream length.
func TestSaveEvents_BatchAssignsSequentialVersions(t *testing.T) {
	store := NewInMemoryEventStore()

	batch := []Event{
		makeTestEvent("agg-1", "Opened", nil),
		makeTestEvent("agg-1", "Deposited", nil),
		makeTestEvent("agg-1", "Withdrawn", nil),
	}

	if err := store.SaveEvents("agg-1", batch, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events, _ := store.LoadEvents("agg-1")
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	for i, evt := range events {
		expectedVersion := i + 1
		if evt.Version != expectedVersion {
			t.Errorf("event %d: expected version %d, got %d",
				i, expectedVersion, evt.Version)
		}
	}
}

// Appending events in successive calls should produce a contiguous
// version sequence across the calls.
func TestSaveEvents_SuccessiveAppends(t *testing.T) {
	store := NewInMemoryEventStore()

	// First save: versions 1-2.
	batch1 := []Event{
		makeTestEvent("agg-1", "Opened", nil),
		makeTestEvent("agg-1", "Deposited", nil),
	}
	if err := store.SaveEvents("agg-1", batch1, 0); err != nil {
		t.Fatalf("batch 1: %v", err)
	}

	// Second save: versions 3-4.
	batch2 := []Event{
		makeTestEvent("agg-1", "Withdrawn", nil),
		makeTestEvent("agg-1", "Deposited", nil),
	}
	if err := store.SaveEvents("agg-1", batch2, 2); err != nil {
		t.Fatalf("batch 2: %v", err)
	}

	events, _ := store.LoadEvents("agg-1")
	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d", len(events))
	}
	for i, evt := range events {
		if evt.Version != i+1 {
			t.Errorf("event %d: expected version %d, got %d",
				i, i+1, evt.Version)
		}
	}
}

// Each event in a single batch should receive a unique ID, even though
// they are saved in the same call.
func TestSaveEvents_UniqueIDs(t *testing.T) {
	store := NewInMemoryEventStore()

	batch := []Event{
		makeTestEvent("agg-1", "A", nil),
		makeTestEvent("agg-1", "B", nil),
		makeTestEvent("agg-1", "C", nil),
	}
	_ = store.SaveEvents("agg-1", batch, 0)

	events, _ := store.LoadEvents("agg-1")
	seen := make(map[string]bool)
	for _, evt := range events {
		if seen[evt.ID] {
			t.Errorf("duplicate event ID: %s", evt.ID)
		}
		seen[evt.ID] = true
	}
}

// Events from different aggregates should be stored independently -
// saving to one aggregate must not affect another.
func TestSaveEvents_IsolatesByAggregateID(t *testing.T) {
	store := NewInMemoryEventStore()

	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 0)
	_ = store.SaveEvents("agg-2", []Event{makeTestEvent("agg-2", "B", nil)}, 0)

	eventsGroup1, _ := store.LoadEvents("agg-1")
	eventsGroup2, _ := store.LoadEvents("agg-2")

	if len(eventsGroup1) != 1 || len(eventsGroup2) != 1 {
		t.Fatalf("expected 1 event per aggregate, got %d and %d",
			len(eventsGroup1), len(eventsGroup2))
	}
	if eventsGroup1[0].EventType != "A" {
		t.Errorf("agg-1 event type: expected 'A', got %q", eventsGroup1[0].EventType)
	}
	if eventsGroup2[0].EventType != "B" {
		t.Errorf("agg-2 event type: expected 'B', got %q", eventsGroup2[0].EventType)
	}
}

// If the expected version does not match the current stream length,
// SaveEvents must return a *ConcurrencyError.
func TestSaveEvents_ConcurrencyConflict(t *testing.T) {
	store := NewInMemoryEventStore()

	// Save one event → stream is now at version 1.
	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 0)

	// Attempt to save expecting version 0 (stale). This simulates a
	// second goroutine that loaded the aggregate before the first save.
	err := store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "B", nil)}, 0)

	if err == nil {
		t.Fatal("expected a concurrency error, got nil")
	}

	var concErr *ConcurrencyError
	if !errors.As(err, &concErr) {
		t.Fatalf("expected *ConcurrencyError, got %T: %v", err, err)
	}
	if concErr.AggregateID != "agg-1" {
		t.Errorf("aggregate ID: expected 'agg-1', got %q", concErr.AggregateID)
	}
	if concErr.ExpectedVersion != 0 {
		t.Errorf("expected version: expected 0, got %d", concErr.ExpectedVersion)
	}
	if concErr.ActualVersion != 1 {
		t.Errorf("actual version: expected 1, got %d", concErr.ActualVersion)
	}
}

// A failed save due to a concurrency conflict must not append any
// events - the stream should remain unchanged.
func TestSaveEvents_ConflictDoesNotMutateStream(t *testing.T) {
	store := NewInMemoryEventStore()

	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 0)

	// This should fail (stale expectedVersion).
	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "B", nil)}, 0)

	events, _ := store.LoadEvents("agg-1")
	if len(events) != 1 {
		t.Errorf("stream should still have 1 event after a failed save, got %d",
			len(events))
	}
}

// An expected version that is higher than the current version should
// also be rejected. This catches defects where a caller miscalculates the
// expected version.
func TestSaveEvents_ExpectedVersionTooHigh(t *testing.T) {
	store := NewInMemoryEventStore()

	err := store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 5)
	if err == nil {
		t.Fatal("expected error when expectedVersion > actual")
	}

	if _, ok := errors.AsType[*ConcurrencyError](err); !ok {
		t.Fatalf("expected *ConcurrencyError, got %T", err)
	}
}

// Loading from a non-existent aggregate should return nil, not an error.
// This is the expected behavior when an aggregate has never been created.
func TestLoadEvents_NonExistentAggregate(t *testing.T) {
	store := NewInMemoryEventStore()

	events, err := store.LoadEvents("does-not-exist")
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if events != nil {
		t.Errorf("expected nil for non-existent aggregate, got %d events",
			len(events))
	}
}

// Events returned by LoadEvents should be in version order, preserving
// the same sequence in which they were saved.
func TestLoadEvents_PreservesOrder(t *testing.T) {
	store := NewInMemoryEventStore()
	saveN(t, store, "agg-1", 5)

	events, _ := store.LoadEvents("agg-1")
	for i := 0; i < len(events)-1; i++ {
		if events[i].Version >= events[i+1].Version {
			t.Errorf("events not in order: version %d at index %d, version %d at index %d",
				events[i].Version, i, events[i+1].Version, i+1)
		}
	}
}

// LoadEvents should return a defensive copy - mutating the returned
// slice must not affect the store's internal data.
func TestLoadEvents_ReturnsCopy(t *testing.T) {
	store := NewInMemoryEventStore()
	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 0)

	// Mutate the returned slice.
	events, _ := store.LoadEvents("agg-1")
	events[0].EventType = "MUTATED"

	// Reload - the store's data should be unaffected.
	fresh, _ := store.LoadEvents("agg-1")
	if fresh[0].EventType == "MUTATED" {
		t.Error("mutating the returned slice should not affect the store")
	}
}

// Loading from version 1 should return the entire stream (same as
// LoadEvents for a complete replay).
func TestLoadEventsFrom_Version1ReturnsAll(t *testing.T) {
	store := NewInMemoryEventStore()
	saveN(t, store, "agg-1", 5)

	events, err := store.LoadEventsFrom("agg-1", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 5 {
		t.Errorf("expected 5 events from version 1, got %d", len(events))
	}
}

// Loading from a mid-stream version should return only the events at
// that version and later.
func TestLoadEventsFrom_MidStream(t *testing.T) {
	store := NewInMemoryEventStore()
	saveN(t, store, "agg-1", 5)

	events, err := store.LoadEventsFrom("agg-1", 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 3 {
		t.Fatalf("expected 3 events from version 3, got %d", len(events))
	}
	if events[0].Version != 3 {
		t.Errorf("first returned event should be version 3, got %d",
			events[0].Version)
	}
	if events[2].Version != 5 {
		t.Errorf("last returned event should be version 5, got %d",
			events[2].Version)
	}
}

// Loading from the very last version should return exactly one event.
func TestLoadEventsFrom_LastVersion(t *testing.T) {
	store := NewInMemoryEventStore()
	saveN(t, store, "agg-1", 5)

	events, _ := store.LoadEventsFrom("agg-1", 5)
	if len(events) != 1 {
		t.Errorf("expected 1 event from the last version, got %d", len(events))
	}
}

// Loading from a version beyond the end of the stream should return nil.
// This happens when a snapshot is at the very latest version and there
// are no newer events to replay.
func TestLoadEventsFrom_BeyondEnd(t *testing.T) {
	store := NewInMemoryEventStore()
	saveN(t, store, "agg-1", 3)

	events, err := store.LoadEventsFrom("agg-1", 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events != nil {
		t.Errorf("expected nil for version beyond end, got %d events",
			len(events))
	}
}

// Loading from a non-existent aggregate should return nil, not an error.
func TestLoadEventsFrom_NonExistentAggregate(t *testing.T) {
	store := NewInMemoryEventStore()

	events, err := store.LoadEventsFrom("ghost", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events != nil {
		t.Errorf("expected nil, got %d events", len(events))
	}
}

// LoadEventsFrom should also return a copy, like LoadEvents.
func TestLoadEventsFrom_ReturnsCopy(t *testing.T) {
	store := NewInMemoryEventStore()
	saveN(t, store, "agg-1", 3)

	events, _ := store.LoadEventsFrom("agg-1", 2)
	events[0].EventType = "MUTATED"

	fresh, _ := store.LoadEventsFrom("agg-1", 2)
	if fresh[0].EventType == "MUTATED" {
		t.Error("mutating the returned slice should not affect the store")
	}
}

// A subscriber registered before events are saved should receive each
// event exactly once, in order.
func TestSubscribe_ReceivesEventsInOrder(t *testing.T) {
	store := NewInMemoryEventStore()

	var received []Event
	store.Subscribe(func(e Event) {
		received = append(received, e)
	})

	batch := []Event{
		makeTestEvent("agg-1", "A", nil),
		makeTestEvent("agg-1", "B", nil),
		makeTestEvent("agg-1", "C", nil),
	}
	_ = store.SaveEvents("agg-1", batch, 0)

	if len(received) != 3 {
		t.Fatalf("subscriber should have received 3 events, got %d",
			len(received))
	}
	for i, evt := range received {
		if evt.Version != i+1 {
			t.Errorf("received[%d]: expected version %d, got %d",
				i, i+1, evt.Version)
		}
	}
}

// Multiple subscribers should all receive every event.
func TestSubscribe_MultipleSubscribers(t *testing.T) {
	store := NewInMemoryEventStore()

	var count1, count2 int
	store.Subscribe(func(e Event) { count1++ })
	store.Subscribe(func(e Event) { count2++ })

	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 0)

	if count1 != 1 {
		t.Errorf("subscriber 1 should have 1 event, got %d", count1)
	}
	if count2 != 1 {
		t.Errorf("subscriber 2 should have 1 event, got %d", count2)
	}
}

// Subscribers should receive the enriched events (with ID, version,
// and timestamp assigned), not the raw input events.
func TestSubscribe_ReceivesEnrichedEvents(t *testing.T) {
	store := NewInMemoryEventStore()

	var received Event
	store.Subscribe(func(e Event) { received = e })

	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "Opened", nil)}, 0)

	if received.ID == "" {
		t.Error("subscriber should receive event with assigned ID")
	}
	if received.Version != 1 {
		t.Errorf("subscriber should see version 1, got %d", received.Version)
	}
	if received.Timestamp.IsZero() {
		t.Error("subscriber should see a non-zero timestamp")
	}
}

// Subscribers should not be notified when a save fails due to a
// concurrency conflict - failed saves must be silent.
func TestSubscribe_NotNotifiedOnConflict(t *testing.T) {
	store := NewInMemoryEventStore()

	callCount := 0
	store.Subscribe(func(e Event) { callCount++ })

	// First save succeeds - 1 notification.
	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 0)

	// Second save fails (stale version) - should not trigger another.
	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "B", nil)}, 0)

	if callCount != 1 {
		t.Errorf("subscriber should have been called once, got %d", callCount)
	}
}

// Events from different aggregates should all flow through the same
// subscriber. Subscribers are global, not per-aggregate.
func TestSubscribe_ReceivesEventsFromAllAggregates(t *testing.T) {
	store := NewInMemoryEventStore()

	var aggregateIDs []string
	store.Subscribe(func(e Event) {
		aggregateIDs = append(aggregateIDs, e.AggregateID)
	})

	_ = store.SaveEvents("agg-1", []Event{makeTestEvent("agg-1", "A", nil)}, 0)
	_ = store.SaveEvents("agg-2", []Event{makeTestEvent("agg-2", "B", nil)}, 0)

	if len(aggregateIDs) != 2 {
		t.Fatalf("expected 2 notifications, got %d", len(aggregateIDs))
	}
	if aggregateIDs[0] != "agg-1" || aggregateIDs[1] != "agg-2" {
		t.Errorf("expected [agg-1, agg-2], got %v", aggregateIDs)
	}
}

// Concurrent saves to different aggregates should all succeed without
// data corruption. This exercises the mutex under contention.
func TestConcurrentSaves_DifferentAggregates(t *testing.T) {
	store := NewInMemoryEventStore()
	const numAggregates = 50

	var wg sync.WaitGroup
	wg.Add(numAggregates)

	for i := 0; i < numAggregates; i++ {
		go func(id string) {
			defer wg.Done()
			evt := makeTestEvent(id, "Created", nil)
			if err := store.SaveEvents(id, []Event{evt}, 0); err != nil {
				t.Errorf("save to %s failed: %v", id, err)
			}
		}(fmt.Sprintf("agg-%d", i))
	}

	wg.Wait()

	// Every aggregate should have exactly 1 event.
	for i := 0; i < numAggregates; i++ {
		id := fmt.Sprintf("agg-%d", i)
		events, _ := store.LoadEvents(id)
		if len(events) != 1 {
			t.Errorf("%s: expected 1 event, got %d", id, len(events))
		}
	}
}

// Saving an empty batch (zero events) with the correct expected version
// should succeed and produce no side effects.
func TestSaveEvents_EmptyBatch(t *testing.T) {
	store := NewInMemoryEventStore()

	err := store.SaveEvents("agg-1", []Event{}, 0)
	if err != nil {
		t.Fatalf("saving an empty batch should succeed, got: %v", err)
	}

	events, _ := store.LoadEvents("agg-1")
	// An empty save to a non-existent aggregate should not create an
	// entry - LoadEvents should still return nil.
	if events != nil {
		t.Errorf("expected nil after empty save, got %d events", len(events))
	}
}

// A freshly created store should have no events for any aggregate.
func TestNewInMemoryEventStore_StartsEmpty(t *testing.T) {
	store := NewInMemoryEventStore()

	events, err := store.LoadEvents("anything")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if events != nil {
		t.Error("new store should return nil for any aggregate")
	}
}
