package account

import (
	"testing"

	"event-sourcing/pkg/es"
)

// openAccountViaStore creates an open account in the event store by saving an AccountOpened event directly.
// This bypasses the command handler so that snapshot tests can focus on
// snapshot specific behavior without re-testing account opening.
func openAccountViaStore(t *testing.T, store es.EventStore, id, owner string, balance float64) {
	t.Helper()
	events := []es.Event{{
		AggregateID:   id,
		AggregateType: AggregateType,
		EventType:     EventAccountOpened,
		Data:          AccountOpenedEvent{Owner: owner, InitialBalance: balance},
	}}
	if err := store.SaveEvents(id, events, 0); err != nil {
		t.Fatalf("failed to seed account %s: %v", id, err)
	}
}

// TestTakeSnapshot_CapturesCurrentState verifies that TakeSnapshot
// produces a snapshot whose fields accurately reflect the aggregate's
// current state. This is the foundation of the snapshotting mechanism:
// if the snapshot does not match the aggregate, restoring from it
// will produce incorrect state.
func TestTakeSnapshot_CapturesCurrentState(t *testing.T) {
	// Given: an account with some history.
	a := NewAccount("acc-001")
	a.LoadFromHistory([]es.Event{
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventAccountOpened,
			Data: AccountOpenedEvent{Owner: "Alice", InitialBalance: 500}},
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyDeposited,
			Data: MoneyDepositedEvent{Amount: 300, Reason: "bonus"}},
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyWithdrawn,
			Data: MoneyWithdrawnEvent{Amount: 100, Reason: "dinner"}},
	})

	// When: we take a snapshot.
	snap := a.TakeSnapshot()

	// Then: the snapshot metadata matches the aggregate.
	if snap.AggregateID != "acc-001" {
		t.Errorf("AggregateID: expected 'acc-001', got %q", snap.AggregateID)
	}
	if snap.AggregateType != AggregateType {
		t.Errorf("AggregateType: expected %q, got %q", AggregateType, snap.AggregateType)
	}
	if snap.Version != 3 {
		t.Errorf("Version: expected 3, got %d", snap.Version)
	}

	// And: the state payload reflects the computed values.
	state, ok := snap.State.(AccountSnapshot)
	if !ok {
		t.Fatalf("expected AccountSnapshot, got %T", snap.State)
	}
	if state.Owner != "Alice" {
		t.Errorf("Owner: expected 'Alice', got %q", state.Owner)
	}
	expectedBalance := 500.0 + 300.0 - 100.0
	if state.Balance != expectedBalance {
		t.Errorf("Balance: expected %.2f, got %.2f", expectedBalance, state.Balance)
	}
	if !state.IsOpen {
		t.Error("IsOpen: expected true, got false")
	}
}

// TestTakeSnapshot_AtVersionZero verifies that taking a snapshot on
// a fresh (unopened) aggregate captures the zero state correctly.
// This is an edge case that should never occur in production (you
// would not snapshot an aggregate with no events), but it exercises
// the boundary condition.
func TestTakeSnapshot_AtVersionZero(t *testing.T) {
	a := NewAccount("acc-empty")

	snap := a.TakeSnapshot()

	if snap.Version != 0 {
		t.Errorf("Version: expected 0, got %d", snap.Version)
	}
	state := snap.State.(AccountSnapshot)
	if state.Owner != "" {
		t.Errorf("Owner: expected empty string, got %q", state.Owner)
	}
	if state.Balance != 0 {
		t.Errorf("Balance: expected 0, got %.2f", state.Balance)
	}
	if state.IsOpen {
		t.Error("IsOpen: expected false for an unopened account")
	}
}

// TestLoadFromSnapshot_RestoresState verifies that an aggregate
// restored from a snapshot has the same observable state as the
// original.
func TestLoadFromSnapshot_RestoresState(t *testing.T) {
	snap := es.Snapshot{
		AggregateID:   "acc-001",
		AggregateType: AggregateType,
		Version:       7,
		State: AccountSnapshot{
			Owner:   "Bob",
			Balance: 1234.56,
			IsOpen:  true,
		},
	}

	restored := NewAccount("acc-001")
	if err := restored.LoadFromSnapshot(snap); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if restored.Owner() != "Bob" {
		t.Errorf("Owner: expected 'Bob', got %q", restored.Owner())
	}
	if restored.Balance() != 1234.56 {
		t.Errorf("Balance: expected 1234.56, got %.2f", restored.Balance())
	}
	if !restored.IsOpen() {
		t.Error("IsOpen: expected true")
	}
	if restored.Version() != 7 {
		t.Errorf("Version: expected 7, got %d", restored.Version())
	}
}

// TestLoadFromSnapshot_InvalidStateType verifies that LoadFromSnapshot
// returns an error when the snapshot's State field does not contain an
// AccountSnapshot. This guards against data corruption or
// deserialization mistakes that produce the wrong type.
func TestLoadFromSnapshot_InvalidStateType(t *testing.T) {
	snap := es.Snapshot{
		AggregateID:   "acc-001",
		AggregateType: AggregateType,
		Version:       3,
		State:         "this is not an AccountSnapshot",
	}

	a := NewAccount("acc-001")
	err := a.LoadFromSnapshot(snap)
	if err == nil {
		t.Fatal("expected an error for invalid state type, got nil")
	}
}

// TestLoadFromSnapshot_ThenReplayEvents verifies the most common
// snapshot usage pattern: load a snapshot to restore state up to
// version N, then replay events N+1..M to bring the aggregate
// fully up to date. The final state must match what a full replay
// from scratch would produce.
func TestLoadFromSnapshot_ThenReplayEvents(t *testing.T) {
	// Simulate a snapshot taken at version 2 (opened + one deposit).
	snap := es.Snapshot{
		AggregateID:   "acc-001",
		AggregateType: AggregateType,
		Version:       2,
		State: AccountSnapshot{
			Owner:   "Carol",
			Balance: 800.0, // opened at 500 + deposited 300
			IsOpen:  true,
		},
	}

	// Replay two more events that happened after the snapshot.
	laterEvents := []es.Event{
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyWithdrawn,
			Data: MoneyWithdrawnEvent{Amount: 200, Reason: "rent"}},
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyDeposited,
			Data: MoneyDepositedEvent{Amount: 50, Reason: "refund"}},
	}

	// Restore from snapshot, then replay.
	a := NewAccount("acc-001")
	if err := a.LoadFromSnapshot(snap); err != nil {
		t.Fatalf("loading snapshot: %v", err)
	}
	a.LoadFromHistory(laterEvents)

	// The version should be snapshot(2) + 2 later events = 4.
	if a.Version() != 4 {
		t.Errorf("Version: expected 4, got %d", a.Version())
	}

	// The balance should be 800 - 200 + 50 = 650.
	expectedBalance := 800.0 - 200.0 + 50.0
	if a.Balance() != expectedBalance {
		t.Errorf("Balance: expected %.2f, got %.2f", expectedBalance, a.Balance())
	}

	// The aggregate should still be usable for new commands.
	if err := a.Withdraw(100, "groceries"); err != nil {
		t.Fatalf("withdrawal after snapshot restore failed: %v", err)
	}
	if a.Balance() != expectedBalance-100.0 {
		t.Errorf("Balance after withdrawal: expected %.2f, got %.2f",
			expectedBalance-100.0, a.Balance())
	}
	if len(a.Changes()) != 1 {
		t.Errorf("Changes: expected 1 uncommitted event, got %d", len(a.Changes()))
	}
}

// TestSnapshotRoundtrip_FullReplayEquivalence builds an aggregate
// two ways - once via full event replay, once via snapshot + partial
// replay - and verifies that both produce identical state. This is
// the ultimate correctness check: the snapshot path must be
// indistinguishable from the full replay path.
func TestSnapshotRoundtrip_FullReplayEquivalence(t *testing.T) {
	allEvents := []es.Event{
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventAccountOpened,
			Data: AccountOpenedEvent{Owner: "Dana", InitialBalance: 1000}},
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyDeposited,
			Data: MoneyDepositedEvent{Amount: 200, Reason: "deposit"}},
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyWithdrawn,
			Data: MoneyWithdrawnEvent{Amount: 350, Reason: "rent"}},
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyDeposited,
			Data: MoneyDepositedEvent{Amount: 75, Reason: "refund"}},
		{AggregateID: "acc-001", AggregateType: AggregateType, EventType: EventMoneyWithdrawn,
			Data: MoneyWithdrawnEvent{Amount: 25, Reason: "fee"}},
	}

	// Path A: full replay.
	fullReplay := NewAccount("acc-001")
	fullReplay.LoadFromHistory(allEvents)

	// Path B: snapshot at version 3, then replay events 4 and 5.
	snapshotAt3 := NewAccount("acc-001")
	snapshotAt3.LoadFromHistory(allEvents[:3])
	snap := snapshotAt3.TakeSnapshot()

	fromSnap := NewAccount("acc-001")
	if err := fromSnap.LoadFromSnapshot(snap); err != nil {
		t.Fatalf("loading snapshot: %v", err)
	}
	fromSnap.LoadFromHistory(allEvents[3:])

	// Both paths should produce identical state.
	if fullReplay.Owner() != fromSnap.Owner() {
		t.Errorf("Owner mismatch: full=%q, snap=%q",
			fullReplay.Owner(), fromSnap.Owner())
	}
	if fullReplay.Balance() != fromSnap.Balance() {
		t.Errorf("Balance mismatch: full=%.2f, snap=%.2f",
			fullReplay.Balance(), fromSnap.Balance())
	}
	if fullReplay.IsOpen() != fromSnap.IsOpen() {
		t.Errorf("IsOpen mismatch: full=%v, snap=%v",
			fullReplay.IsOpen(), fromSnap.IsOpen())
	}
	if fullReplay.Version() != fromSnap.Version() {
		t.Errorf("Version mismatch: full=%d, snap=%d",
			fullReplay.Version(), fromSnap.Version())
	}
}

// TestSnapshotHandler_DepositWithoutSnapshot verifies that the
// snapshot command handler works correctly when no snapshot exists.
// It should fall back to a full event replay, exactly like the
// base command handler.
func TestSnapshotHandler_DepositWithoutSnapshot(t *testing.T) {
	store := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()
	handler := NewSnapshotCommandHandler(store, snapStore, 100)

	// Seed an account via the base handler.
	openAccountViaStore(t, store, "acc-001", "Alice", 500)

	// Deposit using the snapshot handler - no snapshot exists yet.
	if err := handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001", Amount: 200, Reason: "paycheck",
	}); err != nil {
		t.Fatalf("deposit failed: %v", err)
	}

	// Verify state by loading the aggregate from events.
	events, _ := store.LoadEvents("acc-001")
	a := NewAccount("acc-001")
	a.LoadFromHistory(events)

	if a.Balance() != 700.0 {
		t.Errorf("Balance: expected 700.00, got %.2f", a.Balance())
	}
}

// TestSnapshotHandler_WithdrawWithoutSnapshot verifies that the
// snapshot command handler processes withdrawals correctly when
// no snapshot exists.
func TestSnapshotHandler_WithdrawWithoutSnapshot(t *testing.T) {
	store := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()
	handler := NewSnapshotCommandHandler(store, snapStore, 100)

	openAccountViaStore(t, store, "acc-001", "Bob", 1000)

	if err := handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-001", Amount: 300, Reason: "rent",
	}); err != nil {
		t.Fatalf("withdraw failed: %v", err)
	}

	events, _ := store.LoadEvents("acc-001")
	a := NewAccount("acc-001")
	a.LoadFromHistory(events)

	if a.Balance() != 700.0 {
		t.Errorf("Balance: expected 700.00, got %.2f", a.Balance())
	}
}

// TestSnapshotHandler_WithdrawInsufficientFunds verifies that
// business rules (overdraft protection) still work correctly when
// loading through the snapshot path.
func TestSnapshotHandler_WithdrawInsufficientFunds(t *testing.T) {
	store := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()
	handler := NewSnapshotCommandHandler(store, snapStore, 100)

	openAccountViaStore(t, store, "acc-001", "Eve", 100)

	err := handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-001", Amount: 500, Reason: "too-much",
	})
	if err == nil {
		t.Fatal("expected insufficient funds error, got nil")
	}

	// Verify the balance is unchanged - no event was appended.
	events, _ := store.LoadEvents("acc-001")
	if len(events) != 1 {
		t.Errorf("expected only the opening event, got %d events", len(events))
	}
}

// TestSnapshotHandler_SnapshotTakenAtInterval verifies that the
// handler automatically takes a snapshot when the aggregate's
// version reaches a multiple of the configured interval.
func TestSnapshotHandler_SnapshotTakenAtInterval(t *testing.T) {
	store := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()

	// Snapshot every 5 events.
	handler := NewSnapshotCommandHandler(store, snapStore, 5)

	// Event 1: open the account.
	openAccountViaStore(t, store, "acc-001", "Alice", 0)

	// Events 2-4: three deposits via the snapshot handler.
	for i := 0; i < 3; i++ {
		if err := handler.HandleDeposit(DepositCommand{
			AccountID: "acc-001", Amount: 100, Reason: "deposit",
		}); err != nil {
			t.Fatalf("deposit %d failed: %v", i+1, err)
		}
	}

	// At version 4, no snapshot should exist yet.
	snap, _ := snapStore.LoadSnapshot("acc-001")
	if snap != nil {
		t.Fatalf("did not expect a snapshot at version 4, got version %d", snap.Version)
	}

	// Event 5: one more deposit should trigger the snapshot.
	if err := handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001", Amount: 100, Reason: "deposit",
	}); err != nil {
		t.Fatalf("deposit 4 failed: %v", err)
	}

	snap, _ = snapStore.LoadSnapshot("acc-001")
	if snap == nil {
		t.Fatal("expected a snapshot at version 5, got nil")
	}
	if snap.Version != 5 {
		t.Errorf("expected snapshot version 5, got %d", snap.Version)
	}

	// Verify the snapshot's state is correct.
	state, ok := snap.State.(AccountSnapshot)
	if !ok {
		t.Fatalf("expected AccountSnapshot, got %T", snap.State)
	}
	if state.Balance != 400.0 {
		t.Errorf("snapshot balance: expected 400.00, got %.2f", state.Balance)
	}
}

// TestSnapshotHandler_LoadsFromSnapshot verifies that after a
// snapshot is taken, subsequent operations load the aggregate from
// the snapshot rather than replaying all events from the beginning.
// We verify this indirectly by checking that the final state is
// correct after operations that cross the snapshot boundary.
func TestSnapshotHandler_LoadsFromSnapshot(t *testing.T) {
	store := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()

	// Snapshot every 3 events for a tight test loop.
	handler := NewSnapshotCommandHandler(store, snapStore, 3)

	// Event 1: open the account.
	openAccountViaStore(t, store, "acc-001", "Frank", 1000)

	// Events 2-3: two deposits. Event 3 triggers a snapshot.
	for i := 0; i < 2; i++ {
		if err := handler.HandleDeposit(DepositCommand{
			AccountID: "acc-001", Amount: 100, Reason: "deposit",
		}); err != nil {
			t.Fatalf("deposit %d: %v", i+1, err)
		}
	}

	// Confirm the snapshot exists at version 3.
	snap, _ := snapStore.LoadSnapshot("acc-001")
	if snap == nil || snap.Version != 3 {
		t.Fatalf("expected snapshot at version 3, got %v", snap)
	}

	// Events 4-5: two more deposits. These are loaded from
	// snapshot(v3) + events 4-5.
	for i := 0; i < 2; i++ {
		if err := handler.HandleDeposit(DepositCommand{
			AccountID: "acc-001", Amount: 50, Reason: "bonus",
		}); err != nil {
			t.Fatalf("post-snapshot deposit %d: %v", i+1, err)
		}
	}

	// Final balance: 1000 + 100 + 100 + 50 + 50 = 1300.
	events, _ := store.LoadEvents("acc-001")
	a := NewAccount("acc-001")
	a.LoadFromHistory(events)

	if a.Balance() != 1300.0 {
		t.Errorf("Balance: expected 1300.00, got %.2f", a.Balance())
	}
	if a.Version() != 5 {
		t.Errorf("Version: expected 5, got %d", a.Version())
	}
}

// TestSnapshotHandler_ZeroIntervalDisablesSnapshots verifies that
// setting the snapshot interval to 0 disables automatic snapshotting.
// The handler should still work correctly, just without taking any
// snapshots.
func TestSnapshotHandler_ZeroIntervalDisablesSnapshots(t *testing.T) {
	store := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()

	// Interval of 0 means "never snapshot".
	handler := NewSnapshotCommandHandler(store, snapStore, 0)

	openAccountViaStore(t, store, "acc-001", "Heidi", 500)

	// Perform several deposits.
	for i := 0; i < 10; i++ {
		if err := handler.HandleDeposit(DepositCommand{
			AccountID: "acc-001", Amount: 50, Reason: "deposit",
		}); err != nil {
			t.Fatalf("deposit %d: %v", i+1, err)
		}
	}

	// No snapshot should have been taken.
	snap, _ := snapStore.LoadSnapshot("acc-001")
	if snap != nil {
		t.Errorf("expected no snapshot with interval 0, got version %d", snap.Version)
	}

	// But the balance should still be correct.
	events, _ := store.LoadEvents("acc-001")
	a := NewAccount("acc-001")
	a.LoadFromHistory(events)

	if a.Balance() != 1000.0 {
		t.Errorf("Balance: expected 1000.00 (500 + 10*50), got %.2f", a.Balance())
	}
}

// TestSnapshotHandler_HandlesDepositOnNonExistentAccount verifies
// that the snapshot handler correctly returns an error when
// attempting to deposit into an account that was never opened.
// The account is loaded via snapshot path (no snapshot found) then
// via full replay (no events found), yielding an empty aggregate
// that rejects the deposit.
func TestSnapshotHandler_HandlesDepositOnNonExistentAccount(t *testing.T) {
	store := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()
	handler := NewSnapshotCommandHandler(store, snapStore, 5)

	err := handler.HandleDeposit(DepositCommand{
		AccountID: "ghost-account", Amount: 100, Reason: "test",
	})
	if err == nil {
		t.Fatal("expected an error when depositing to a non-existent account")
	}
}
