package account

import (
	"fmt"

	"event-sourcing/pkg/es"
)

// AccountSnapshot represents the serializable state of an account
// at a specific point in its event stream. This is what gets stored
// in the snapshot store.
type AccountSnapshot struct {
	Owner   string  `json:"owner"`
	Balance float64 `json:"balance"`
	IsOpen  bool    `json:"is_open"`
}

// TakeSnapshot creates a snapshot of the account's current state.
func (a *Account) TakeSnapshot() es.Snapshot {
	return es.Snapshot{
		AggregateID:   a.id,
		AggregateType: AggregateType,
		Version:       a.version,
		State: AccountSnapshot{
			Owner:   a.owner,
			Balance: a.balance,
			IsOpen:  a.isOpen,
		},
	}
}

// LoadFromSnapshot restores the aggregate's state from a snapshot.
// After calling this, the aggregate's version matches the snapshot's
// version, and subsequent events can be replayed from that point.
func (a *Account) LoadFromSnapshot(snapshot es.Snapshot) error {
	state, ok := snapshot.State.(AccountSnapshot)
	if !ok {
		return fmt.Errorf("invalid snapshot state type: %T", snapshot.State)
	}

	a.owner = state.Owner
	a.balance = state.Balance
	a.isOpen = state.IsOpen
	a.version = snapshot.Version
	return nil
}

// SnapshotCommandHandler extends CommandHandler with snapshot support.
// It loads aggregates from snapshots when available, falling back to
// full event replay when no snapshot exists. It also takes snapshots
// at configurable intervals.
type SnapshotCommandHandler struct {
	store        es.EventStore
	snapStore    es.SnapshotStore
	snapInterval int // take a snapshot every N events
}

// NewSnapshotCommandHandler creates a handler with snapshot support.
// The snapInterval parameter controls how often snapshots are taken.
// A typical value is 50-100 events.
func NewSnapshotCommandHandler(
	store es.EventStore,
	snapStore es.SnapshotStore,
	snapInterval int,
) *SnapshotCommandHandler {
	return &SnapshotCommandHandler{
		store:        store,
		snapStore:    snapStore,
		snapInterval: snapInterval,
	}
}

// loadAccountWithSnapshot loads an aggregate using the most efficient
// strategy: snapshot + recent events if a snapshot exists, or full
// replay if not.
func (h *SnapshotCommandHandler) loadAccountWithSnapshot(id string) (*Account, error) {
	acct := NewAccount(id)

	// Try to load a snapshot first.
	snapshot, err := h.snapStore.LoadSnapshot(id)
	if err != nil {
		return nil, fmt.Errorf("loading snapshot: %w", err)
	}

	if snapshot != nil {
		// Snapshot found: restore state and load only events after it.
		if err = acct.LoadFromSnapshot(*snapshot); err != nil {
			return nil, fmt.Errorf("applying snapshot: %w", err)
		}

		events, err := h.store.LoadEventsFrom(id, snapshot.Version+1)
		if err != nil {
			return nil, fmt.Errorf(
				"loading events from version %d: %w",
				snapshot.Version+1, err,
			)
		}
		acct.LoadFromHistory(events)
	} else {
		// No snapshot: full replay from the beginning.
		events, err := h.store.LoadEvents(id)
		if err != nil {
			return nil, fmt.Errorf("loading events: %w", err)
		}
		acct.LoadFromHistory(events)
	}

	return acct, nil
}

// saveChangesWithSnapshot persists events and takes a snapshot if
// the aggregate has accumulated enough events since the last one.
func (h *SnapshotCommandHandler) saveChangesWithSnapshot(acct *Account) error {
	expectedVersion := acct.Version() - len(acct.Changes())
	if err := h.store.SaveEvents(acct.ID(), acct.Changes(), expectedVersion); err != nil {
		return err
	}

	// Check if it is time for a new snapshot.
	if h.snapInterval > 0 && acct.Version()%h.snapInterval == 0 {
		snapshot := acct.TakeSnapshot()
		if err := h.snapStore.SaveSnapshot(snapshot); err != nil {
			// Snapshot failure is not fatal - the system works
			// without it, just slower. Log the error in production.
			return nil
		}
	}

	return nil
}

// HandleDeposit processes a deposit using snapshot accelerated loading.
func (h *SnapshotCommandHandler) HandleDeposit(cmd DepositCommand) error {
	acct, err := h.loadAccountWithSnapshot(cmd.AccountID)
	if err != nil {
		return err
	}

	if err = acct.Deposit(cmd.Amount, cmd.Reason); err != nil {
		return err
	}

	return h.saveChangesWithSnapshot(acct)
}

// HandleWithdraw processes a withdrawal using snapshot accelerated
// loading.
func (h *SnapshotCommandHandler) HandleWithdraw(cmd WithdrawCommand) error {
	acct, err := h.loadAccountWithSnapshot(cmd.AccountID)
	if err != nil {
		return err
	}

	if err = acct.Withdraw(cmd.Amount, cmd.Reason); err != nil {
		return err
	}

	return h.saveChangesWithSnapshot(acct)
}
