package account

import (
	"strings"
	"testing"

	"event-sourcing/pkg/es"
)

// openAccount returns an Account that has been opened with the given
// owner and initial balance, ready for further operations. This is the
// most common "given" setup: most commands require an open account.
func openAccount(id, owner string, balance float64) *Account {
	a := NewAccount(id)
	a.LoadFromHistory([]es.Event{
		{
			AggregateID:   id,
			AggregateType: AggregateType,
			EventType:     EventAccountOpened,
			Data: AccountOpenedEvent{
				Owner:          owner,
				InitialBalance: balance,
			},
		},
	})
	return a
}

// makeEvent builds a minimal Event envelope. The store normally assigns
// ID, Version, and Timestamp, but the aggregate's apply method does not
// depend on those fields - it only reads Data and increments version.
func makeEvent(aggregateID, eventType string, data any) es.Event {
	return es.Event{
		AggregateID:   aggregateID,
		AggregateType: AggregateType,
		EventType:     eventType,
		Data:          data,
	}
}

// A freshly constructed account should have the provided ID and be in
// a completely blank state: not open, zero balance, zero version, no
// owner, and no uncommitted changes.
func TestNewAccount_BlankState(t *testing.T) {
	a := NewAccount("acc-001")

	if a.ID() != "acc-001" {
		t.Errorf("expected ID 'acc-001', got %q", a.ID())
	}
	if a.Owner() != "" {
		t.Errorf("expected empty owner, got %q", a.Owner())
	}
	if a.Balance() != 0 {
		t.Errorf("expected zero balance, got %.2f", a.Balance())
	}
	if a.IsOpen() {
		t.Error("expected account to not be open")
	}
	if a.Version() != 0 {
		t.Errorf("expected version 0, got %d", a.Version())
	}
	if len(a.Changes()) != 0 {
		t.Errorf("expected no changes, got %d", len(a.Changes()))
	}
}

// Opening a fresh account with a positive initial balance should set
// the owner, balance, and isOpen flag, produce exactly one uncommitted
// event of type AccountOpened, and advance the version to 1.
func TestOpenAccount_Success_WithBalance(t *testing.T) {
	a := NewAccount("acc-001")

	err := a.OpenAccount("Alice", 250.0)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if a.Owner() != "Alice" {
		t.Errorf("expected owner 'Alice', got %q", a.Owner())
	}
	if a.Balance() != 250.0 {
		t.Errorf("expected balance 250.00, got %.2f", a.Balance())
	}
	if !a.IsOpen() {
		t.Error("expected account to be open")
	}
	if a.Version() != 1 {
		t.Errorf("expected version 1, got %d", a.Version())
	}
	if len(a.Changes()) != 1 {
		t.Fatalf("expected 1 change, got %d", len(a.Changes()))
	}
}

// Opening an account with a zero initial balance is valid. Many real
// accounts start empty and receive their first deposit later.
func TestOpenAccount_Success_ZeroBalance(t *testing.T) {
	a := NewAccount("acc-001")

	err := a.OpenAccount("Bob", 0)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if a.Balance() != 0 {
		t.Errorf("expected balance 0, got %.2f", a.Balance())
	}
	if !a.IsOpen() {
		t.Error("expected account to be open")
	}
}

// Attempting to open an already-open account must be rejected. The
// event stream for an account can only contain one AccountOpened event,
// and the aggregate enforces this invariant.
func TestOpenAccount_AlreadyOpen(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	err := a.OpenAccount("Bob", 50.0)

	if err == nil {
		t.Fatal("expected error when opening already-open account")
	}
	// The state should remain completely unchanged.
	if a.Owner() != "Alice" {
		t.Errorf("owner should remain 'Alice', got %q", a.Owner())
	}
	if len(a.Changes()) != 0 {
		t.Errorf("expected 0 changes after rejection, got %d", len(a.Changes()))
	}
}

// An account requires an owner name. An empty string should be rejected.
func TestOpenAccount_EmptyOwner(t *testing.T) {
	a := NewAccount("acc-001")

	err := a.OpenAccount("", 100.0)

	if err == nil {
		t.Fatal("expected error for empty owner")
	}
	if a.IsOpen() {
		t.Error("account should not be open after rejected command")
	}
	if len(a.Changes()) != 0 {
		t.Errorf("expected 0 changes, got %d", len(a.Changes()))
	}
}

// A negative initial balance makes no sense for a new account and
// should be rejected.
func TestOpenAccount_NegativeBalance(t *testing.T) {
	a := NewAccount("acc-001")

	err := a.OpenAccount("Alice", -50.0)

	if err == nil {
		t.Fatal("expected error for negative initial balance")
	}
	if a.IsOpen() {
		t.Error("account should not be open after rejected command")
	}
}

// The event produced by OpenAccount should carry the correct envelope
// metadata: AggregateID, AggregateType, EventType, and the domain
// payload with owner and initial balance.
func TestOpenAccount_EventMetadata(t *testing.T) {
	a := NewAccount("acc-007")
	_ = a.OpenAccount("James", 777.0)

	if len(a.Changes()) != 1 {
		t.Fatalf("expected 1 change, got %d", len(a.Changes()))
	}

	evt := a.Changes()[0]

	if evt.AggregateID != "acc-007" {
		t.Errorf("event AggregateID: expected 'acc-007', got %q", evt.AggregateID)
	}
	if evt.AggregateType != AggregateType {
		t.Errorf("event AggregateType: expected %q, got %q",
			AggregateType, evt.AggregateType)
	}
	if evt.EventType != EventAccountOpened {
		t.Errorf("event EventType: expected %q, got %q",
			EventAccountOpened, evt.EventType)
	}

	data, ok := evt.Data.(AccountOpenedEvent)
	if !ok {
		t.Fatalf("expected AccountOpenedEvent, got %T", evt.Data)
	}
	if data.Owner != "James" {
		t.Errorf("event data Owner: expected 'James', got %q", data.Owner)
	}
	if data.InitialBalance != 777.0 {
		t.Errorf("event data InitialBalance: expected 777.0, got %.2f",
			data.InitialBalance)
	}
}

// A valid deposit on an open account should increase the balance,
// produce one uncommitted MoneyDeposited event, and advance the version.
func TestDeposit_Success(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	err := a.Deposit(50.0, "paycheck")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if a.Balance() != 150.0 {
		t.Errorf("expected balance 150.00, got %.2f", a.Balance())
	}
	if a.Version() != 2 {
		t.Errorf("expected version 2, got %d", a.Version())
	}
	if len(a.Changes()) != 1 {
		t.Fatalf("expected 1 change, got %d", len(a.Changes()))
	}

	data, ok := a.Changes()[0].Data.(MoneyDepositedEvent)
	if !ok {
		t.Fatalf("expected MoneyDepositedEvent, got %T", a.Changes()[0].Data)
	}
	if data.Amount != 50.0 {
		t.Errorf("expected amount 50.0, got %.2f", data.Amount)
	}
	if data.Reason != "paycheck" {
		t.Errorf("expected reason 'paycheck', got %q", data.Reason)
	}
}

// When the caller provides an empty reason string, the aggregate should
// default it to "deposit" so the event stream is always self-documenting.
func TestDeposit_DefaultReason(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	_ = a.Deposit(25.0, "")

	data := a.Changes()[0].Data.(MoneyDepositedEvent)
	if data.Reason != "deposit" {
		t.Errorf("expected default reason 'deposit', got %q", data.Reason)
	}
}

// Depositing into an account that has not been opened should be rejected.
// This protects against commands arriving before the account exists.
func TestDeposit_AccountNotOpen(t *testing.T) {
	a := NewAccount("acc-001")

	err := a.Deposit(50.0, "test")

	if err == nil {
		t.Fatal("expected error for unopened account")
	}
	if len(a.Changes()) != 0 {
		t.Errorf("expected 0 changes, got %d", len(a.Changes()))
	}
}

// A zero deposit makes no real-world sense and should be rejected.
func TestDeposit_ZeroAmount(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	err := a.Deposit(0, "zero")

	if err == nil {
		t.Fatal("expected error for zero deposit")
	}
	if a.Balance() != 100.0 {
		t.Errorf("balance should be unchanged at 100.00, got %.2f", a.Balance())
	}
}

// A negative deposit should be rejected. Negative amounts should flow
// through the Withdraw path, not be smuggled through Deposit.
func TestDeposit_NegativeAmount(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	err := a.Deposit(-25.0, "sneaky")

	if err == nil {
		t.Fatal("expected error for negative deposit")
	}
	if a.Balance() != 100.0 {
		t.Errorf("balance should be unchanged at 100.00, got %.2f", a.Balance())
	}
}

// The deposit event should carry the correct envelope metadata.
func TestDeposit_EventMetadata(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)
	_ = a.Deposit(75.0, "bonus")

	evt := a.Changes()[0]
	if evt.AggregateID != "acc-001" {
		t.Errorf("AggregateID: expected 'acc-001', got %q", evt.AggregateID)
	}
	if evt.AggregateType != AggregateType {
		t.Errorf("AggregateType: expected %q, got %q",
			AggregateType, evt.AggregateType)
	}
	if evt.EventType != EventMoneyDeposited {
		t.Errorf("EventType: expected %q, got %q",
			EventMoneyDeposited, evt.EventType)
	}
}

// A valid withdrawal on an open account with sufficient funds should
// decrease the balance, produce one MoneyWithdrawn event, and advance
// the version.
func TestWithdraw_Success(t *testing.T) {
	a := openAccount("acc-001", "Alice", 200.0)

	err := a.Withdraw(75.0, "groceries")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if a.Balance() != 125.0 {
		t.Errorf("expected balance 125.00, got %.2f", a.Balance())
	}
	if a.Version() != 2 {
		t.Errorf("expected version 2, got %d", a.Version())
	}
	if len(a.Changes()) != 1 {
		t.Fatalf("expected 1 change, got %d", len(a.Changes()))
	}

	data, ok := a.Changes()[0].Data.(MoneyWithdrawnEvent)
	if !ok {
		t.Fatalf("expected MoneyWithdrawnEvent, got %T", a.Changes()[0].Data)
	}
	if data.Amount != 75.0 {
		t.Errorf("expected amount 75.0, got %.2f", data.Amount)
	}
	if data.Reason != "groceries" {
		t.Errorf("expected reason 'groceries', got %q", data.Reason)
	}
}

// Withdrawing the exact balance should succeed. This is the boundary
// case - the balance drops to exactly zero, which is allowed.
func TestWithdraw_ExactBalance(t *testing.T) {
	a := openAccount("acc-001", "Alice", 300.0)

	err := a.Withdraw(300.0, "close-out")

	if err != nil {
		t.Fatalf("withdrawing exact balance should succeed: %v", err)
	}
	if a.Balance() != 0 {
		t.Errorf("expected balance 0.00, got %.2f", a.Balance())
	}
}

// When the caller provides an empty reason, the aggregate should
// default it to "withdrawal".
func TestWithdraw_DefaultReason(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	_ = a.Withdraw(10.0, "")

	data := a.Changes()[0].Data.(MoneyWithdrawnEvent)
	if data.Reason != "withdrawal" {
		t.Errorf("expected default reason 'withdrawal', got %q", data.Reason)
	}
}

// The overdraft protection rule: withdrawing more than the current
// balance must be rejected, and no event should be produced.
func TestWithdraw_InsufficientBalance(t *testing.T) {
	a := openAccount("acc-001", "Alice", 50.0)

	err := a.Withdraw(100.0, "rent")

	if err == nil {
		t.Fatal("expected error for insufficient balance")
	}
	if a.Balance() != 50.0 {
		t.Errorf("balance should remain 50.00, got %.2f", a.Balance())
	}
	if len(a.Changes()) != 0 {
		t.Errorf("expected 0 changes after rejection, got %d", len(a.Changes()))
	}
	if a.Version() != 1 {
		t.Errorf("version should remain 1, got %d", a.Version())
	}
}

// Withdrawing from an unopened account should be rejected.
func TestWithdraw_AccountNotOpen(t *testing.T) {
	a := NewAccount("acc-001")

	err := a.Withdraw(10.0, "test")

	if err == nil {
		t.Fatal("expected error for unopened account")
	}
}

// A zero withdrawal should be rejected.
func TestWithdraw_ZeroAmount(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	err := a.Withdraw(0, "zero")

	if err == nil {
		t.Fatal("expected error for zero withdrawal")
	}
	if a.Balance() != 100.0 {
		t.Errorf("balance should be unchanged at 100.00, got %.2f", a.Balance())
	}
}

// A negative withdrawal should be rejected.
func TestWithdraw_NegativeAmount(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	err := a.Withdraw(-10.0, "negative")

	if err == nil {
		t.Fatal("expected error for negative withdrawal")
	}
}

// The withdrawal event should carry the correct envelope metadata.
func TestWithdraw_EventMetadata(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)
	_ = a.Withdraw(20.0, "coffee")

	evt := a.Changes()[0]
	if evt.AggregateID != "acc-001" {
		t.Errorf("AggregateID: expected 'acc-001', got %q", evt.AggregateID)
	}
	if evt.AggregateType != AggregateType {
		t.Errorf("AggregateType: expected %q, got %q",
			AggregateType, evt.AggregateType)
	}
	if evt.EventType != EventMoneyWithdrawn {
		t.Errorf("EventType: expected %q, got %q",
			EventMoneyWithdrawn, evt.EventType)
	}
}

// Replaying events from an empty slice should leave the aggregate in
// its blank initial state. This is the case when the event store has
// no events for the aggregate - it has never been used.
func TestLoadFromHistory_EmptySlice(t *testing.T) {
	a := NewAccount("acc-001")

	a.LoadFromHistory([]es.Event{})

	if a.IsOpen() {
		t.Error("account should not be open after replaying zero events")
	}
	if a.Version() != 0 {
		t.Errorf("expected version 0, got %d", a.Version())
	}
}

// Replaying a nil event slice should behave identically to an empty
// slice - the aggregate stays blank, and there is no panic.
func TestLoadFromHistory_NilSlice(t *testing.T) {
	a := NewAccount("acc-001")

	a.LoadFromHistory(nil)

	if a.IsOpen() {
		t.Error("account should not be open")
	}
	if a.Version() != 0 {
		t.Errorf("expected version 0, got %d", a.Version())
	}
}

// Replaying historical events should rebuild the aggregate's state
// correctly and should not produce any uncommitted changes. The
// distinction between "historical replay" and "new command" is the
// core of the dual-path apply mechanism.
func TestLoadFromHistory_DoesNotProduceChanges(t *testing.T) {
	a := NewAccount("acc-001")

	a.LoadFromHistory([]es.Event{
		makeEvent("acc-001", EventAccountOpened, AccountOpenedEvent{
			Owner: "Alice", InitialBalance: 500,
		}),
		makeEvent("acc-001", EventMoneyDeposited, MoneyDepositedEvent{
			Amount: 200, Reason: "deposit",
		}),
	})

	if len(a.Changes()) != 0 {
		t.Errorf("replaying history should produce 0 changes, got %d",
			len(a.Changes()))
	}
	// State should still be correct despite no changes.
	if a.Balance() != 700.0 {
		t.Errorf("expected balance 700.00, got %.2f", a.Balance())
	}
	if a.Version() != 2 {
		t.Errorf("expected version 2, got %d", a.Version())
	}
}

// A full sequence of mixed events should produce the correct
// cumulative state. This tests the "left fold" property: the
// aggregate's state is the result of applying every event in order.
func TestLoadFromHistory_CumulativeState(t *testing.T) {
	a := NewAccount("acc-001")

	a.LoadFromHistory([]es.Event{
		makeEvent("acc-001", EventAccountOpened, AccountOpenedEvent{
			Owner: "Alice", InitialBalance: 0,
		}),
		makeEvent("acc-001", EventMoneyDeposited, MoneyDepositedEvent{
			Amount: 1000, Reason: "paycheck",
		}),
		makeEvent("acc-001", EventMoneyWithdrawn, MoneyWithdrawnEvent{
			Amount: 200, Reason: "rent",
		}),
		makeEvent("acc-001", EventMoneyDeposited, MoneyDepositedEvent{
			Amount: 50, Reason: "refund",
		}),
		makeEvent("acc-001", EventMoneyWithdrawn, MoneyWithdrawnEvent{
			Amount: 100, Reason: "groceries",
		}),
	})

	// 0 + 1000 - 200 + 50 - 100 = 750
	if a.Balance() != 750.0 {
		t.Errorf("expected balance 750.00, got %.2f", a.Balance())
	}
	if a.Owner() != "Alice" {
		t.Errorf("expected owner 'Alice', got %q", a.Owner())
	}
	if !a.IsOpen() {
		t.Error("expected account to be open")
	}
	if a.Version() != 5 {
		t.Errorf("expected version 5, got %d", a.Version())
	}
}

// An event with an unrecognized Data type should not crash the
// aggregate. The apply method's type switch will simply not match,
// but the version counter should still increment. This is the
// "tolerant reader" approach - unknown events are silently skipped
// in terms of state, but still counted.
func TestLoadFromHistory_UnknownEventType(t *testing.T) {
	a := NewAccount("acc-001")

	a.LoadFromHistory([]es.Event{
		makeEvent("acc-001", EventAccountOpened, AccountOpenedEvent{
			Owner: "Alice", InitialBalance: 100,
		}),
		makeEvent("acc-001", "SomeFutureEvent", struct{ Foo string }{"bar"}),
	})

	// The unknown event should not affect balance or owner, but
	// version should still count it.
	if a.Balance() != 100.0 {
		t.Errorf("expected balance 100.00, got %.2f", a.Balance())
	}
	if a.Version() != 2 {
		t.Errorf("expected version 2 (unknown event still counted), got %d",
			a.Version())
	}
}

// After loading from history, issuing a new command should produce
// changes that reflect only the new command - the historical events
// should not appear in Changes(). The version should reflect both
// the historical events and the new command.
func TestHistoryThenCommand_OnlyNewChanges(t *testing.T) {
	a := NewAccount("acc-001")
	a.LoadFromHistory([]es.Event{
		makeEvent("acc-001", EventAccountOpened, AccountOpenedEvent{
			Owner: "Alice", InitialBalance: 500,
		}),
		makeEvent("acc-001", EventMoneyDeposited, MoneyDepositedEvent{
			Amount: 200, Reason: "deposit",
		}),
	})

	// The account now has version 2, balance 700, and 0 changes.
	_ = a.Deposit(100.0, "bonus")

	if len(a.Changes()) != 1 {
		t.Fatalf("expected 1 change (only the new deposit), got %d",
			len(a.Changes()))
	}
	if a.Changes()[0].EventType != EventMoneyDeposited {
		t.Errorf("expected %s, got %s",
			EventMoneyDeposited, a.Changes()[0].EventType)
	}
	if a.Version() != 3 {
		t.Errorf("expected version 3 (2 history + 1 new), got %d", a.Version())
	}
	if a.Balance() != 800.0 {
		t.Errorf("expected balance 800.00, got %.2f", a.Balance())
	}
}

// Issuing multiple commands after loading from history should
// accumulate all new events in Changes() while keeping historical
// events out.
func TestHistoryThenMultipleCommands(t *testing.T) {
	a := openAccount("acc-001", "Alice", 1000.0)

	_ = a.Deposit(200.0, "bonus")
	_ = a.Withdraw(50.0, "lunch")
	_ = a.Deposit(30.0, "refund")

	if len(a.Changes()) != 3 {
		t.Fatalf("expected 3 changes, got %d", len(a.Changes()))
	}

	// Verify the event types appear in the correct order.
	expectedTypes := []string{
		EventMoneyDeposited,
		EventMoneyWithdrawn,
		EventMoneyDeposited,
	}
	for i, want := range expectedTypes {
		got := a.Changes()[i].EventType
		if got != want {
			t.Errorf("change[%d]: expected %s, got %s", i, want, got)
		}
	}

	// 1000 + 200 - 50 + 30 = 1180
	if a.Balance() != 1180.0 {
		t.Errorf("expected balance 1180.00, got %.2f", a.Balance())
	}
	if a.Version() != 4 {
		t.Errorf("expected version 4, got %d", a.Version())
	}
}

// A rejected command in the middle of a sequence should not produce
// an event, should not change state, but the prior successful
// commands should be preserved in Changes().
func TestHistoryThenPartialFailure(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	// First command succeeds.
	err := a.Deposit(50.0, "deposit")
	if err != nil {
		t.Fatalf("deposit should succeed: %v", err)
	}

	// Second command fails (insufficient funds: have 150, need 500).
	err = a.Withdraw(500.0, "too-much")
	if err == nil {
		t.Fatal("expected error for insufficient balance")
	}

	// Third command succeeds.
	err = a.Withdraw(30.0, "coffee")
	if err != nil {
		t.Fatalf("small withdrawal should succeed: %v", err)
	}

	// Only the two successful commands should appear in changes.
	if len(a.Changes()) != 2 {
		t.Fatalf("expected 2 changes (failed command excluded), got %d",
			len(a.Changes()))
	}

	// 100 + 50 - 30 = 120
	if a.Balance() != 120.0 {
		t.Errorf("expected balance 120.00, got %.2f", a.Balance())
	}
	// version: 1 (history) + 2 (successful commands) = 3
	// The failed command does NOT increment the version.
	if a.Version() != 3 {
		t.Errorf("expected version 3, got %d", a.Version())
	}
}

// ClearChanges should remove all uncommitted events. This is called
// by the command handler after events have been successfully persisted
// to the event store.
func TestClearChanges(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)
	_ = a.Deposit(50.0, "test")
	_ = a.Withdraw(20.0, "test")

	if len(a.Changes()) != 2 {
		t.Fatalf("expected 2 changes before clear, got %d", len(a.Changes()))
	}

	a.ClearChanges()

	if a.Changes() != nil {
		t.Errorf("expected nil changes after clear, got %v", a.Changes())
	}

	// State should be unaffected by clearing changes.
	if a.Balance() != 130.0 {
		t.Errorf("expected balance 130.00 (unchanged), got %.2f", a.Balance())
	}
	if a.Version() != 3 {
		t.Errorf("expected version 3 (unchanged), got %d", a.Version())
	}
}

// After clearing changes and issuing a new command, only the new
// command should appear in Changes(). This simulates the pattern
// of "persist, clear, then handle the next command."
func TestClearChanges_ThenNewCommand(t *testing.T) {
	a := openAccount("acc-001", "Alice", 100.0)

	_ = a.Deposit(50.0, "first")
	a.ClearChanges()
	_ = a.Deposit(25.0, "second")

	if len(a.Changes()) != 1 {
		t.Fatalf("expected 1 change after clear+new, got %d", len(a.Changes()))
	}
	data := a.Changes()[0].Data.(MoneyDepositedEvent)
	if data.Amount != 25.0 {
		t.Errorf("expected amount 25.0, got %.2f", data.Amount)
	}
}

// The version counter should increment by exactly 1 for each event
// applied, whether the event comes from history or from a command.
// This counter is what the event store uses for optimistic concurrency.
func TestVersionTracking(t *testing.T) {
	a := NewAccount("acc-001")

	if a.Version() != 0 {
		t.Fatalf("fresh account should be version 0, got %d", a.Version())
	}

	_ = a.OpenAccount("Alice", 100.0)
	if a.Version() != 1 {
		t.Errorf("after open: expected version 1, got %d", a.Version())
	}

	_ = a.Deposit(50.0, "d1")
	if a.Version() != 2 {
		t.Errorf("after deposit: expected version 2, got %d", a.Version())
	}

	_ = a.Withdraw(10.0, "w1")
	if a.Version() != 3 {
		t.Errorf("after withdraw: expected version 3, got %d", a.Version())
	}

	_ = a.Deposit(20.0, "d2")
	if a.Version() != 4 {
		t.Errorf("after second deposit: expected version 4, got %d", a.Version())
	}
}

// Error messages should include the account ID so that when multiple
// accounts are processed concurrently, the failing account can be
// identified from the error alone.
func TestErrorMessages_IncludeAccountID(t *testing.T) {
	tests := []struct {
		name string
		fn   func() error
	}{
		{
			name: "deposit on unopened account",
			fn: func() error {
				a := NewAccount("acc-XYZ")
				return a.Deposit(10, "test")
			},
		},
		{
			name: "withdraw from unopened account",
			fn: func() error {
				a := NewAccount("acc-XYZ")
				return a.Withdraw(10, "test")
			},
		},
		{
			name: "re-open already open account",
			fn: func() error {
				a := openAccount("acc-XYZ", "Alice", 100)
				return a.OpenAccount("Bob", 0)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.fn()
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), "acc-XYZ") {
				t.Errorf("error %q should contain the account ID 'acc-XYZ'",
					err.Error())
			}
		})
	}
}

// The aggregate's state should be identical whether it was built from
// a single LoadFromHistory call or from individual commands. This
// property is what makes event sourcing reliable: replaying events
// always produces the same state as processing them live.
func TestStateDerivedFromHistory_MatchesLiveCommands(t *testing.T) {
	// Path 1: build state by issuing commands one by one.
	live := NewAccount("acc-001")
	_ = live.OpenAccount("Alice", 0)
	_ = live.Deposit(500.0, "paycheck")
	_ = live.Withdraw(120.0, "rent")
	_ = live.Deposit(30.0, "refund")

	// Capture the events that were produced.
	events := live.Changes()

	// Path 2: build state by replaying those same events as history.
	replayed := NewAccount("acc-001")
	replayed.LoadFromHistory(events)

	// Both should produce identical state.
	if replayed.Owner() != live.Owner() {
		t.Errorf("owner mismatch: live=%q, replayed=%q",
			live.Owner(), replayed.Owner())
	}
	if replayed.Balance() != live.Balance() {
		t.Errorf("balance mismatch: live=%.2f, replayed=%.2f",
			live.Balance(), replayed.Balance())
	}
	if replayed.IsOpen() != live.IsOpen() {
		t.Errorf("isOpen mismatch: live=%v, replayed=%v",
			live.IsOpen(), replayed.IsOpen())
	}
	if replayed.Version() != live.Version() {
		t.Errorf("version mismatch: live=%d, replayed=%d",
			live.Version(), replayed.Version())
	}

	// The replayed aggregate should have zero changes (they are
	// historical), while the live aggregate has all of them.
	if len(replayed.Changes()) != 0 {
		t.Errorf("replayed should have 0 changes, got %d",
			len(replayed.Changes()))
	}
	if len(live.Changes()) != 4 {
		t.Errorf("live should have 4 changes, got %d",
			len(live.Changes()))
	}
}
