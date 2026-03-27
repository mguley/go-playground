package projection

import (
	"testing"

	"event-sourcing/pkg/account"
	"event-sourcing/pkg/es"
)

// saveEvent is a convenience that wraps a domain event payload into
// an es.Event envelope and persists it through the store. This
// triggers subscriber notifications, which is how projections
// receive events in production. Using the store (rather than calling
// handleEvent directly) ensures our tests exercise the same wiring
// the real system uses.
func saveEvent(t *testing.T, store *es.InMemoryEventStore, aggregateID string, eventType string, data any, expectedVersion int) {
	t.Helper()
	err := store.SaveEvents(aggregateID, []es.Event{
		{
			AggregateID:   aggregateID,
			AggregateType: account.AggregateType,
			EventType:     eventType,
			Data:          data,
		},
	}, expectedVersion)
	if err != nil {
		t.Fatalf("failed to save event %s for %s: %v", eventType, aggregateID, err)
	}
}

// TestBalanceProjection_AccountOpened verifies that opening an
// account creates a view with the correct owner, balance, and status.
func TestBalanceProjection_AccountOpened(t *testing.T) {
	// Given: an empty event store with a balance projection subscribed.
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	// When: an AccountOpened event is persisted.
	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 250}, 0)

	// Then: the projection holds the correct view.
	view := projection.GetAccount("acc-001")
	if view == nil {
		t.Fatal("expected non-nil view after AccountOpened")
	}
	if view.AccountID != "acc-001" {
		t.Errorf("expected account ID 'acc-001', got %q", view.AccountID)
	}
	if view.Owner != "Alice" {
		t.Errorf("expected owner 'Alice', got %q", view.Owner)
	}
	if view.Balance != 250 {
		t.Errorf("expected balance 250, got %.2f", view.Balance)
	}
	if !view.IsOpen {
		t.Error("expected account to be marked as open")
	}
	if view.Version != 1 {
		t.Errorf("expected version 1, got %d", view.Version)
	}
}

// TestBalanceProjection_AccountOpenedWithZeroBalance ensures that an
// account opened with no initial deposit is tracked correctly.
func TestBalanceProjection_AccountOpenedWithZeroBalance(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Bob", InitialBalance: 0}, 0)

	view := projection.GetAccount("acc-001")
	if view == nil {
		t.Fatal("expected non-nil view")
	}
	if view.Balance != 0 {
		t.Errorf("expected balance 0, got %.2f", view.Balance)
	}
}

// TestBalanceProjection_DepositUpdatesBalance verifies that a
// deposit event increments the projected balance.
func TestBalanceProjection_DepositUpdatesBalance(t *testing.T) {
	// Given: an open account with $100.
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)
	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)

	// When: $50 is deposited.
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 50, Reason: "paycheck"}, 1)

	// Then: the balance reflects the deposit.
	view := projection.GetAccount("acc-001")
	if view == nil {
		t.Fatal("expected non-nil view")
	}
	if view.Balance != 150 {
		t.Errorf("expected balance 150, got %.2f", view.Balance)
	}
	if view.Version != 2 {
		t.Errorf("expected version 2, got %d", view.Version)
	}
}

// TestBalanceProjection_WithdrawalUpdatesBalance verifies that a
// withdrawal event decrements the projected balance.
func TestBalanceProjection_WithdrawalUpdatesBalance(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)
	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 500}, 0)

	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 200, Reason: "rent"}, 1)

	view := projection.GetAccount("acc-001")
	if view == nil {
		t.Fatal("expected non-nil view")
	}
	if view.Balance != 300 {
		t.Errorf("expected balance 300, got %.2f", view.Balance)
	}
}

// TestBalanceProjection_MultipleOperations verifies that a sequence
// of deposits and withdrawals produces the correct cumulative
// balance in the projection.
func TestBalanceProjection_MultipleOperations(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	// Open with $0, deposit $1000, withdraw $200, deposit $50, withdraw $75.
	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 0}, 0)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 1000, Reason: "paycheck"}, 1)
	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 200, Reason: "rent"}, 2)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 50, Reason: "refund"}, 3)
	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 75, Reason: "groceries"}, 4)

	// Expected: 0 + 1000 - 200 + 50 - 75 = 775
	view := projection.GetAccount("acc-001")
	if view == nil {
		t.Fatal("expected non-nil view")
	}
	if view.Balance != 775 {
		t.Errorf("expected balance 775, got %.2f", view.Balance)
	}
	if view.Version != 5 {
		t.Errorf("expected version 5, got %d", view.Version)
	}
}

// TestBalanceProjection_GetAccountReturnsNilForUnknown verifies that
// querying a nonexistent account returns nil rather than panicking.
func TestBalanceProjection_GetAccountReturnsNilForUnknown(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	view := projection.GetAccount("no-such-account")
	if view != nil {
		t.Errorf("expected nil for unknown account, got %+v", view)
	}
}

// TestBalanceProjection_GetAccountReturnsCopy verifies that the
// returned view is a copy: mutating it does not affect the
// projection's internal state. This is important for thread safety
// and data integrity.
func TestBalanceProjection_GetAccountReturnsCopy(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)
	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)

	// Get a view and mutate it.
	view := projection.GetAccount("acc-001")
	if view == nil {
		t.Fatal("expected non-nil view")
	}

	view.Balance = 999_999

	// The projection's internal data should be untouched.
	fresh := projection.GetAccount("acc-001")
	if fresh == nil {
		t.Fatal("expected non-nil fresh")
	}
	if fresh.Balance != 100 {
		t.Errorf("expected internal balance to remain 100, got %.2f", fresh.Balance)
	}
}

// TestBalanceProjection_MultipleAccounts verifies that the projection
// tracks each account independently.
func TestBalanceProjection_MultipleAccounts(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 1000}, 0)
	saveEvent(t, store, "acc-002", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Bob", InitialBalance: 500}, 0)

	// Deposit into Alice only.
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 200, Reason: "bonus"}, 1)

	alice := projection.GetAccount("acc-001")
	bob := projection.GetAccount("acc-002")

	if alice == nil {
		t.Fatal("expected non-nil alice projection")
	}
	if bob == nil {
		t.Fatal("expected non-nil bob projection")
	}

	if alice.Balance != 1200 {
		t.Errorf("alice: expected 1200, got %.2f", alice.Balance)
	}
	if bob.Balance != 500 {
		t.Errorf("bob: expected 500 (untouched), got %.2f", bob.Balance)
	}
}

// TestBalanceProjection_GetAllAccounts verifies that all accounts
// are returned and sorted by account ID.
func TestBalanceProjection_GetAllAccounts(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	// Open accounts in non-alphabetical order to test sorting.
	saveEvent(t, store, "acc-charlie", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Charlie", InitialBalance: 300}, 0)
	saveEvent(t, store, "acc-alice", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)
	saveEvent(t, store, "acc-bob", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Bob", InitialBalance: 200}, 0)

	all := projection.GetAllAccounts()
	if len(all) != 3 {
		t.Fatalf("expected 3 accounts, got %d", len(all))
	}

	// They should come back sorted by AccountID.
	expectedOrder := []string{"acc-alice", "acc-bob", "acc-charlie"}
	for i, expected := range expectedOrder {
		if all[i].AccountID != expected {
			t.Errorf("position %d: expected %s, got %s", i, expected, all[i].AccountID)
		}
	}
}

// TestBalanceProjection_GetAllAccountsEmpty verifies that an empty
// projection returns an empty (not nil) slice.
func TestBalanceProjection_GetAllAccountsEmpty(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	all := projection.GetAllAccounts()
	if all == nil {
		t.Error("expected non-nil empty slice, got nil")
	}
	if len(all) != 0 {
		t.Errorf("expected 0 accounts, got %d", len(all))
	}
}

// TestBalanceProjection_IgnoresOtherAggregateTypes verifies that
// events belonging to a different aggregate type (not "Account")
// are silently ignored by the projection.
func TestBalanceProjection_IgnoresOtherAggregateTypes(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewBalanceProjection(store)

	// Save an event with a different aggregate type directly.
	err := store.SaveEvents("other-001", []es.Event{
		{
			AggregateID:   "other-001",
			AggregateType: "SomeOtherAggregate",
			EventType:     account.EventMoneyDeposited,
			Data:          account.MoneyDepositedEvent{Amount: 999, Reason: "test"},
		},
	}, 0)
	if err != nil {
		t.Fatalf("save: %v", err)
	}

	// The projection should not have created any view.
	view := projection.GetAccount("other-001")
	if view != nil {
		t.Errorf("expected nil for non-Account aggregate, got %+v", view)
	}
}
