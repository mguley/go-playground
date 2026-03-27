package es

import (
	"sync"
	"testing"
)

// TestSaveAndLoadSnapshot verifies the basic round-trip: saving a
// snapshot and loading it back produces the same data.
func TestSaveAndLoadSnapshot(t *testing.T) {
	store := NewInMemorySnapshotStore()

	// Given: a snapshot representing an account at version 3.
	original := Snapshot{
		AggregateID:   "acc-001",
		AggregateType: "Account",
		Version:       3,
		State: map[string]any{
			"owner":   "Alice",
			"balance": 600.0,
			"is_open": true,
		},
	}

	// When: we save and reload the snapshot.
	if err := store.SaveSnapshot(original); err != nil {
		t.Fatalf("unexpected error saving snapshot: %v", err)
	}

	loaded, err := store.LoadSnapshot("acc-001")
	if err != nil {
		t.Fatalf("unexpected error loading snapshot: %v", err)
	}

	// Then: the loaded snapshot matches the original on all
	// identifier and version fields, and has a non-zero timestamp.
	if loaded == nil {
		t.Fatal("expected a snapshot, got nil")
	}
	if loaded.AggregateID != original.AggregateID {
		t.Errorf("AggregateID: expected %q, got %q",
			original.AggregateID, loaded.AggregateID)
	}
	if loaded.AggregateType != original.AggregateType {
		t.Errorf("AggregateType: expected %q, got %q",
			original.AggregateType, loaded.AggregateType)
	}
	if loaded.Version != original.Version {
		t.Errorf("Version: expected %d, got %d",
			original.Version, loaded.Version)
	}
	if loaded.Timestamp.IsZero() {
		t.Error("expected Timestamp to be set, but it is zero")
	}
}

// TestLoadSnapshot_NonExistent verifies that loading a snapshot for
// an aggregate that has never been snapshotted returns nil without
// an error. This is the expected case when an aggregate is loaded
// for the first time - the command handler should fall back to a
// full event replay.
func TestLoadSnapshot_NonExistent(t *testing.T) {
	store := NewInMemorySnapshotStore()

	loaded, err := store.LoadSnapshot("does-not-exist")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil for non-existent snapshot, got %+v", loaded)
	}
}

// TestSaveSnapshot_OverwritesPrevious verifies that saving a new
// snapshot for the same aggregate replaces the old one. The snapshot
// store only keeps the most recent snapshot per aggregate - older
// snapshots are discarded because they are redundant once a newer
// one exists.
func TestSaveSnapshot_OverwritesPrevious(t *testing.T) {
	store := NewInMemorySnapshotStore()

	// Save an initial snapshot at version 5.
	first := Snapshot{
		AggregateID:   "acc-001",
		AggregateType: "Account",
		Version:       5,
		State:         "state-at-v5",
	}
	if err := store.SaveSnapshot(first); err != nil {
		t.Fatalf("saving first snapshot: %v", err)
	}

	// Save a newer snapshot at version 10.
	second := Snapshot{
		AggregateID:   "acc-001",
		AggregateType: "Account",
		Version:       10,
		State:         "state-at-v10",
	}
	if err := store.SaveSnapshot(second); err != nil {
		t.Fatalf("saving second snapshot: %v", err)
	}

	// The loaded snapshot should be the second one, not the first.
	loaded, err := store.LoadSnapshot("acc-001")
	if err != nil {
		t.Fatalf("loading snapshot: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected a snapshot, got nil")
	}
	if loaded.Version != 10 {
		t.Errorf("expected version 10 (latest snapshot), got %d", loaded.Version)
	}
	if loaded.State != "state-at-v10" {
		t.Errorf("expected state from second snapshot, got %v", loaded.State)
	}
}

// TestSaveSnapshot_SetsTimestamp verifies that SaveSnapshot assigns
// a timestamp automatically, even if the caller did not set one.
// This ensures every snapshot has a meaningful timestamp for
// debugging and monitoring purposes.
func TestSaveSnapshot_SetsTimestamp(t *testing.T) {
	store := NewInMemorySnapshotStore()

	snap := Snapshot{
		AggregateID:   "acc-001",
		AggregateType: "Account",
		Version:       1,
		State:         "some-state",
		// Timestamp intentionally left as zero value.
	}

	if err := store.SaveSnapshot(snap); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	loaded, err := store.LoadSnapshot("acc-001")
	if err != nil {
		t.Fatalf("unexpected error loading: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected a snapshot, got nil")
	}
	if loaded.Timestamp.IsZero() {
		t.Error("expected SaveSnapshot to assign a non-zero timestamp")
	}
}

// TestSaveSnapshot_IsolatedAggregates verifies that snapshots for
// different aggregates are stored independently. Saving a snapshot
// for aggregate A must not affect aggregate B's snapshot.
func TestSaveSnapshot_IsolatedAggregates(t *testing.T) {
	store := NewInMemorySnapshotStore()

	snapA := Snapshot{
		AggregateID:   "acc-alice",
		AggregateType: "Account",
		Version:       3,
		State:         "alice-state",
	}
	snapB := Snapshot{
		AggregateID:   "acc-bob",
		AggregateType: "Account",
		Version:       7,
		State:         "bob-state",
	}

	if err := store.SaveSnapshot(snapA); err != nil {
		t.Fatalf("saving alice snapshot: %v", err)
	}
	if err := store.SaveSnapshot(snapB); err != nil {
		t.Fatalf("saving bob snapshot: %v", err)
	}

	// Each aggregate should return its own snapshot.
	loadedA, _ := store.LoadSnapshot("acc-alice")
	loadedB, _ := store.LoadSnapshot("acc-bob")

	if loadedA.Version != 3 {
		t.Errorf("alice: expected version 3, got %d", loadedA.Version)
	}
	if loadedB.Version != 7 {
		t.Errorf("bob: expected version 7, got %d", loadedB.Version)
	}
	if loadedA.State != "alice-state" {
		t.Errorf("alice: expected state 'alice-state', got %v", loadedA.State)
	}
	if loadedB.State != "bob-state" {
		t.Errorf("bob: expected state 'bob-state', got %v", loadedB.State)
	}
}

// TestSnapshotStore_ConcurrentAccess verifies that the snapshot
// store is safe for concurrent use. Multiple goroutines reading and
// writing simultaneously should not cause data races or panics.
func TestSnapshotStore_ConcurrentAccess(t *testing.T) {
	store := NewInMemorySnapshotStore()
	const numGoroutines = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2) // half writers, half readers

	// Spawn writers: each saves a snapshot with a different version.
	for i := 0; i < numGoroutines; i++ {
		go func(version int) {
			defer wg.Done()
			snap := Snapshot{
				AggregateID:   "acc-001",
				AggregateType: "Account",
				Version:       version,
				State:         version,
			}
			if err := store.SaveSnapshot(snap); err != nil {
				t.Errorf("concurrent save failed: %v", err)
			}
		}(i)
	}

	// Spawn readers: each loads the snapshot (which may or may not
	// exist yet, depending on goroutine scheduling).
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_, err := store.LoadSnapshot("acc-001")
			if err != nil {
				t.Errorf("concurrent load failed: %v", err)
			}
		}()
	}

	wg.Wait()

	// After all goroutines finish, the snapshot should exist and
	// have a valid version (we do not know which writer won the
	// last-write-wins race, but the version should be in range).
	final, err := store.LoadSnapshot("acc-001")
	if err != nil {
		t.Fatalf("final load failed: %v", err)
	}
	if final == nil {
		t.Fatal("expected a snapshot after concurrent writes, got nil")
	}
	if final.Version < 0 || final.Version >= numGoroutines {
		t.Errorf("unexpected final version %d (expected 0..%d)",
			final.Version, numGoroutines-1)
	}
}
