package account

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"event-sourcing/pkg/es"
)

// setupHandler creates a fresh event store and command handler, which
// is the minimal infrastructure needed to test any command. Every test
// starts from this baseline.
func setupHandler() (*es.InMemoryEventStore, *CommandHandler) {
	store := es.NewInMemoryEventStore()
	handler := NewCommandHandler(store)
	return store, handler
}

// openTestAccount is a shorthand that opens an account through the
// command handler so that subsequent commands (deposit, withdraw,
// transfer) have a valid target. It fails the test immediately if
// the open operation returns an error.
func openTestAccount(t *testing.T, handler *CommandHandler, id, owner string, balance float64) {
	t.Helper()
	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      id,
		Owner:          owner,
		InitialBalance: balance,
	})
	if err != nil {
		t.Fatalf("failed to open account %s: %v", id, err)
	}
}

// accountBalance loads an account's events from the store, replays
// them, and returns the resulting balance. This verifies the actual
// persisted state rather than relying on projections.
func accountBalance(t *testing.T, store *es.InMemoryEventStore, id string) float64 {
	t.Helper()
	events, err := store.LoadEvents(id)
	if err != nil {
		t.Fatalf("loading events for %s: %v", id, err)
	}
	acct := NewAccount(id)
	acct.LoadFromHistory(events)
	return acct.Balance()
}

// accountVersion loads an account and returns its version (the number
// of events in its stream). Useful for verifying that the expected
// number of events were persisted.
func accountVersion(t *testing.T, store *es.InMemoryEventStore, id string) int {
	t.Helper()
	events, err := store.LoadEvents(id)
	if err != nil {
		t.Fatalf("loading events for %s: %v", id, err)
	}
	return len(events)
}

// Opening a new account should persist exactly one event and produce
// the correct balance when the events are replayed.
func TestHandleOpenAccount_Success(t *testing.T) {
	store, handler := setupHandler()

	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      "acc-001",
		Owner:          "Alice",
		InitialBalance: 500.0,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the event was persisted by replaying from the store.
	if accountBalance(t, store, "acc-001") != 500.0 {
		t.Errorf("expected balance 500.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
	if accountVersion(t, store, "acc-001") != 1 {
		t.Errorf("expected 1 event in store, got %d",
			accountVersion(t, store, "acc-001"))
	}
}

// Opening an account with zero initial balance should succeed. Many
// accounts start empty and receive their first deposit later.
func TestHandleOpenAccount_ZeroBalance(t *testing.T) {
	store, handler := setupHandler()

	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      "acc-001",
		Owner:          "Alice",
		InitialBalance: 0,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if accountBalance(t, store, "acc-001") != 0 {
		t.Errorf("expected balance 0.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// The handler should reject an empty account ID before even touching
// the event store. This is handler level validation, not aggregate
// level - the aggregate never sees the command.
func TestHandleOpenAccount_EmptyAccountID(t *testing.T) {
	_, handler := setupHandler()

	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      "",
		Owner:          "Alice",
		InitialBalance: 100,
	})

	if err == nil {
		t.Fatal("expected error for empty account ID")
	}
}

// Opening the same account twice should fail. The handler loads the
// existing events, rebuilds the aggregate, and the aggregate's own
// "already open" check rejects the command.
func TestHandleOpenAccount_AlreadyExists(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 100)

	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      "acc-001",
		Owner:          "Bob",
		InitialBalance: 50,
	})

	if err == nil {
		t.Fatal("expected error when opening already-open account")
	}

	// The store should still have exactly 1 event (the original open).
	if accountVersion(t, store, "acc-001") != 1 {
		t.Errorf("store should still have 1 event, got %d",
			accountVersion(t, store, "acc-001"))
	}
}

// An empty owner should be rejected by the aggregate's validation.
// The handler should propagate this error cleanly.
func TestHandleOpenAccount_EmptyOwner(t *testing.T) {
	_, handler := setupHandler()

	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      "acc-001",
		Owner:          "",
		InitialBalance: 100,
	})

	if err == nil {
		t.Fatal("expected error for empty owner")
	}
}

// A negative initial balance should be rejected.
func TestHandleOpenAccount_NegativeBalance(t *testing.T) {
	_, handler := setupHandler()

	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      "acc-001",
		Owner:          "Alice",
		InitialBalance: -50,
	})

	if err == nil {
		t.Fatal("expected error for negative initial balance")
	}
}

// When a validation error causes the open to fail, no events should
// be persisted to the store. This confirms that the handler only
// calls SaveEvents after the aggregate succeeds.
func TestHandleOpenAccount_FailureDoesNotPersist(t *testing.T) {
	store, handler := setupHandler()

	// This fails because owner is empty.
	_ = handler.HandleOpenAccount(OpenAccountCommand{
		AccountID:      "acc-001",
		Owner:          "",
		InitialBalance: 100,
	})

	events, _ := store.LoadEvents("acc-001")
	if events != nil {
		t.Errorf("no events should be persisted after a failed open, got %d",
			len(events))
	}
}

// Depositing into an open account should increase its persisted balance.
func TestHandleDeposit_Success(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 200)

	err := handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001",
		Amount:    150,
		Reason:    "paycheck",
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if accountBalance(t, store, "acc-001") != 350.0 {
		t.Errorf("expected balance 350.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
	// 1 open event + 1 deposit event = 2 events total.
	if accountVersion(t, store, "acc-001") != 2 {
		t.Errorf("expected 2 events, got %d",
			accountVersion(t, store, "acc-001"))
	}
}

// Multiple deposits should accumulate correctly. This tests that
// the handler re-loads the aggregate's full history each time,
// so each deposit sees the correct current balance.
func TestHandleDeposit_Multiple(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 0)

	for i := 0; i < 5; i++ {
		err := handler.HandleDeposit(DepositCommand{
			AccountID: "acc-001",
			Amount:    100,
			Reason:    fmt.Sprintf("deposit-%d", i+1),
		})
		if err != nil {
			t.Fatalf("deposit %d: %v", i+1, err)
		}
	}

	// 0 + 5*100 = 500
	if accountBalance(t, store, "acc-001") != 500.0 {
		t.Errorf("expected balance 500.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
	// 1 open + 5 deposits = 6 events
	if accountVersion(t, store, "acc-001") != 6 {
		t.Errorf("expected 6 events, got %d",
			accountVersion(t, store, "acc-001"))
	}
}

// Depositing into an account that has never been opened should fail.
// The handler loads zero events, the aggregate stays in its blank
// state (isOpen=false), and the Deposit call rejects the command.
func TestHandleDeposit_AccountNotOpen(t *testing.T) {
	_, handler := setupHandler()

	err := handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001",
		Amount:    100,
		Reason:    "test",
	})

	if err == nil {
		t.Fatal("expected error for deposit on non-existent account")
	}
}

// A zero amount deposit should be rejected by the aggregate and
// the handler should propagate the error without persisting anything.
func TestHandleDeposit_ZeroAmount(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 100)

	err := handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001",
		Amount:    0,
		Reason:    "zero",
	})

	if err == nil {
		t.Fatal("expected error for zero deposit")
	}
	// Balance should be unchanged - still just the initial 100.
	if accountBalance(t, store, "acc-001") != 100.0 {
		t.Errorf("balance should be unchanged at 100.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// A negative deposit should be rejected.
func TestHandleDeposit_NegativeAmount(t *testing.T) {
	_, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 100)

	err := handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001",
		Amount:    -50,
		Reason:    "negative",
	})

	if err == nil {
		t.Fatal("expected error for negative deposit")
	}
}

// The deposit event should persist the reason provided by the caller.
// This is verified by replaying the events and inspecting the last one.
func TestHandleDeposit_PersistsReason(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 0)

	_ = handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001",
		Amount:    100,
		Reason:    "birthday-gift",
	})

	events, _ := store.LoadEvents("acc-001")
	// The last event should be the deposit.
	lastEvent := events[len(events)-1]
	data, ok := lastEvent.Data.(MoneyDepositedEvent)
	if !ok {
		t.Fatalf("last event should be MoneyDepositedEvent, got %T",
			lastEvent.Data)
	}
	if data.Reason != "birthday-gift" {
		t.Errorf("expected reason 'birthday-gift', got %q", data.Reason)
	}
}

// A valid withdrawal should reduce the persisted balance.
func TestHandleWithdraw_Success(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 500)

	err := handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-001",
		Amount:    200,
		Reason:    "rent",
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if accountBalance(t, store, "acc-001") != 300.0 {
		t.Errorf("expected balance 300.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// Withdrawing more than the available balance should fail. The
// aggregate's overdraft protection should reject the command, and
// no event should be persisted.
func TestHandleWithdraw_InsufficientFunds(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 100)

	err := handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-001",
		Amount:    500,
		Reason:    "too-much",
	})

	if err == nil {
		t.Fatal("expected error for insufficient funds")
	}
	// Balance should be unchanged.
	if accountBalance(t, store, "acc-001") != 100.0 {
		t.Errorf("balance should remain 100.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
	// Only the open event should exist - the failed withdrawal
	// should not have been persisted.
	if accountVersion(t, store, "acc-001") != 1 {
		t.Errorf("expected 1 event (just the open), got %d",
			accountVersion(t, store, "acc-001"))
	}
}

// Withdrawing from a non-existent account should fail.
func TestHandleWithdraw_AccountNotOpen(t *testing.T) {
	_, handler := setupHandler()

	err := handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-001",
		Amount:    50,
		Reason:    "test",
	})

	if err == nil {
		t.Fatal("expected error for withdrawal on non-existent account")
	}
}

// Zero and negative withdrawals should be rejected.
func TestHandleWithdraw_InvalidAmounts(t *testing.T) {
	_, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 100)

	tests := []struct {
		name   string
		amount float64
	}{
		{"zero", 0},
		{"negative", -10},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := handler.HandleWithdraw(WithdrawCommand{
				AccountID: "acc-001",
				Amount:    tc.amount,
				Reason:    "test",
			})
			if err == nil {
				t.Errorf("expected error for %s withdrawal", tc.name)
			}
		})
	}
}

// Withdrawing the exact balance should succeed, leaving a zero balance.
func TestHandleWithdraw_ExactBalance(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 250)

	err := handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-001",
		Amount:    250,
		Reason:    "close-out",
	})

	if err != nil {
		t.Fatalf("withdrawing exact balance should succeed: %v", err)
	}
	if accountBalance(t, store, "acc-001") != 0 {
		t.Errorf("expected balance 0.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// After a successful deposit and withdrawal sequence, the persisted
// state should reflect the cumulative result. This verifies that
// each handler invocation correctly loads the latest state from
// the store before processing.
func TestHandleDepositThenWithdraw_Sequence(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 100)

	_ = handler.HandleDeposit(DepositCommand{
		AccountID: "acc-001", Amount: 300, Reason: "bonus",
	})
	_ = handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-001", Amount: 150, Reason: "shopping",
	})

	// 100 + 300 - 150 = 250
	if accountBalance(t, store, "acc-001") != 250.0 {
		t.Errorf("expected balance 250.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// A valid transfer should debit the source and credit the destination.
func TestHandleTransfer_Success(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 1000)
	openTestAccount(t, handler, "acc-002", "Bob", 200)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-002",
		Amount:        300,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if accountBalance(t, store, "acc-001") != 700.0 {
		t.Errorf("alice: expected 700.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
	if accountBalance(t, store, "acc-002") != 500.0 {
		t.Errorf("bob: expected 500.00, got %.2f",
			accountBalance(t, store, "acc-002"))
	}
}

// Transferring to the same account is nonsensical and should be
// caught by the handler's own validation, before any aggregate
// is loaded or any event is produced.
func TestHandleTransfer_SameAccount(t *testing.T) {
	_, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 1000)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-001",
		Amount:        100,
	})

	if err == nil {
		t.Fatal("expected error for same-account transfer")
	}
}

// A zero transfer amount should be rejected.
func TestHandleTransfer_ZeroAmount(t *testing.T) {
	_, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 1000)
	openTestAccount(t, handler, "acc-002", "Bob", 200)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-002",
		Amount:        0,
	})

	if err == nil {
		t.Fatal("expected error for zero transfer amount")
	}
}

// A negative transfer amount should be rejected.
func TestHandleTransfer_NegativeAmount(t *testing.T) {
	_, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 1000)
	openTestAccount(t, handler, "acc-002", "Bob", 200)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-002",
		Amount:        -50,
	})

	if err == nil {
		t.Fatal("expected error for negative transfer amount")
	}
}

// A transfer that exceeds the source's balance should fail at the
// withdrawal step. Because the withdrawal is rejected before it is
// persisted, no events should be produced on either account.
func TestHandleTransfer_InsufficientFunds(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 100)
	openTestAccount(t, handler, "acc-002", "Bob", 200)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-002",
		Amount:        500,
	})

	if err == nil {
		t.Fatal("expected error for insufficient funds on transfer")
	}

	// Both balances should be unchanged because the withdrawal failed
	// before any event was persisted.
	if accountBalance(t, store, "acc-001") != 100.0 {
		t.Errorf("alice: expected 100.00 (unchanged), got %.2f",
			accountBalance(t, store, "acc-001"))
	}
	if accountBalance(t, store, "acc-002") != 200.0 {
		t.Errorf("bob: expected 200.00 (unchanged), got %.2f",
			accountBalance(t, store, "acc-002"))
	}
}

// Transferring from a non-existent account should fail at the load
// step. The error should mention the source account.
func TestHandleTransfer_SourceNotOpen(t *testing.T) {
	_, handler := setupHandler()
	openTestAccount(t, handler, "acc-002", "Bob", 200)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-ghost",
		ToAccountID:   "acc-002",
		Amount:        100,
	})

	if err == nil {
		t.Fatal("expected error when source account doesn't exist")
	}
}

// Transferring to a non-existent account should fail at the deposit
// step. The withdrawal on the source has already been persisted at
// this point, so the source account is debited - this is the
// documented limitation that a Saga/process manager would fix.
func TestHandleTransfer_DestinationNotOpen(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 500)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-ghost",
		Amount:        100,
	})

	if err == nil {
		t.Fatal("expected error when destination account doesn't exist")
	}

	// The source account's withdrawal was persisted (the handler
	// commits the withdrawal before attempting the deposit). This
	// is the known limitation documented in the HandleTransfer code.
	// In a production system, a Saga would emit a compensating event.
	if accountBalance(t, store, "acc-001") != 400.0 {
		t.Errorf("alice: expected 400.00 (debited), got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// The transfer should produce "transfer_out" and "transfer_in" as
// the event reasons, making the event stream self-documenting. This
// test replays both accounts' events and inspects the last event
// on each.
func TestHandleTransfer_EventReasons(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 1000)
	openTestAccount(t, handler, "acc-002", "Bob", 200)

	_ = handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-002",
		Amount:        300,
	})

	// Check the source account's last event.
	aliceEvents, _ := store.LoadEvents("acc-001")
	lastAlice := aliceEvents[len(aliceEvents)-1]
	withdrawal, ok := lastAlice.Data.(MoneyWithdrawnEvent)
	if !ok {
		t.Fatalf("source last event: expected MoneyWithdrawnEvent, got %T",
			lastAlice.Data)
	}
	if withdrawal.Reason != "transfer_out" {
		t.Errorf("source reason: expected 'transfer_out', got %q",
			withdrawal.Reason)
	}
	if withdrawal.Amount != 300.0 {
		t.Errorf("source amount: expected 300.00, got %.2f", withdrawal.Amount)
	}

	// Check the destination account's last event.
	bobEvents, _ := store.LoadEvents("acc-002")
	lastBob := bobEvents[len(bobEvents)-1]
	deposit, ok := lastBob.Data.(MoneyDepositedEvent)
	if !ok {
		t.Fatalf("dest last event: expected MoneyDepositedEvent, got %T",
			lastBob.Data)
	}
	if deposit.Reason != "transfer_in" {
		t.Errorf("dest reason: expected 'transfer_in', got %q",
			deposit.Reason)
	}
	if deposit.Amount != 300.0 {
		t.Errorf("dest amount: expected 300.00, got %.2f", deposit.Amount)
	}
}

// Multiple transfers between the same accounts should accumulate
// correctly, demonstrating that the handler reloads the latest
// state from the store on each invocation.
func TestHandleTransfer_MultipleTransfers(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 1000)
	openTestAccount(t, handler, "acc-002", "Bob", 0)

	// Transfer 200 three times: Alice to Bob.
	for i := 0; i < 3; i++ {
		err := handler.HandleTransfer(TransferCommand{
			FromAccountID: "acc-001",
			ToAccountID:   "acc-002",
			Amount:        200,
		})
		if err != nil {
			t.Fatalf("transfer %d: %v", i+1, err)
		}
	}

	// Alice: 1000 - 3*200 = 400
	// Bob:   0 + 3*200 = 600
	if accountBalance(t, store, "acc-001") != 400.0 {
		t.Errorf("alice: expected 400.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
	if accountBalance(t, store, "acc-002") != 600.0 {
		t.Errorf("bob: expected 600.00, got %.2f",
			accountBalance(t, store, "acc-002"))
	}
}

// Each command handler invocation must load the full current state
// from the store. This test verifies that the handler does not cache
// stale state by performing operations from two separate handler
// instances that share the same store.
func TestPipeline_HandlersShareStoreState(t *testing.T) {
	store := es.NewInMemoryEventStore()
	handler1 := NewCommandHandler(store)
	handler2 := NewCommandHandler(store)

	// handler1 opens the account.
	openTestAccount(t, handler1, "acc-001", "Alice", 500)

	// handler2 deposits - it should see the account that handler1 opened.
	err := handler2.HandleDeposit(DepositCommand{
		AccountID: "acc-001",
		Amount:    100,
		Reason:    "from-handler2",
	})

	if err != nil {
		t.Fatalf("handler2 deposit failed: %v", err)
	}
	if accountBalance(t, store, "acc-001") != 600.0 {
		t.Errorf("expected balance 600.00, got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// Optimistic concurrency: if two handlers load the same aggregate
// version simultaneously, the second save should fail. This test
// manually simulates the race by loading events, creating two
// aggregates from the same snapshot, and saving in sequence.
func TestPipeline_OptimisticConcurrencyOnSameAccount(t *testing.T) {
	store, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 1000)

	// Both "handlers" load the same version (1 event = version 1).
	events, _ := store.LoadEvents("acc-001")

	acct1 := NewAccount("acc-001")
	acct1.LoadFromHistory(events)

	acct2 := NewAccount("acc-001")
	acct2.LoadFromHistory(events)

	// Both attempt to withdraw $800.
	_ = acct1.Withdraw(800, "withdrawal-1")
	_ = acct2.Withdraw(800, "withdrawal-2")

	// First save succeeds.
	err := store.SaveEvents("acc-001", acct1.Changes(), 1)
	if err != nil {
		t.Fatalf("first save should succeed: %v", err)
	}

	// Second save should fail with a concurrency error.
	err = store.SaveEvents("acc-001", acct2.Changes(), 1)
	if err == nil {
		t.Fatal("expected concurrency error on second save")
	}

	if _, ok := errors.AsType[*es.ConcurrencyError](err); !ok {
		t.Fatalf("expected *ConcurrencyError, got %T: %v", err, err)
	}

	// Only one withdrawal should have been persisted. The balance
	// should be 1000 - 800 = 200, not 1000 - 800 - 800 = -600.
	if accountBalance(t, store, "acc-001") != 200.0 {
		t.Errorf("expected balance 200.00 (one withdrawal), got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}

// The transfer handler wraps errors from the source withdrawal step
// so that callers can distinguish where in the transfer the failure
// occurred.
func TestHandleTransfer_ErrorContextFromSource(t *testing.T) {
	_, handler := setupHandler()
	openTestAccount(t, handler, "acc-001", "Alice", 50)
	openTestAccount(t, handler, "acc-002", "Bob", 200)

	err := handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-001",
		ToAccountID:   "acc-002",
		Amount:        500,
	})

	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "source") {
		t.Errorf("error should mention 'source' for context: %q", err.Error())
	}
}

// A complex workflow mixing deposits, withdrawals, and transfers
// should produce a consistent final state. This exercises every
// handler method in combination.
func TestCrossOperation_ComplexWorkflow(t *testing.T) {
	store, handler := setupHandler()

	// Open three accounts.
	openTestAccount(t, handler, "acc-A", "Alice", 1000)
	openTestAccount(t, handler, "acc-B", "Bob", 500)
	openTestAccount(t, handler, "acc-C", "Carol", 200)

	// Alice deposits 200 (balance: 1200).
	_ = handler.HandleDeposit(DepositCommand{
		AccountID: "acc-A", Amount: 200, Reason: "bonus",
	})

	// Transfer 300 from Alice to Bob (A: 900, B: 800).
	_ = handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-A", ToAccountID: "acc-B", Amount: 300,
	})

	// Bob withdraws 100 (B: 700).
	_ = handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-B", Amount: 100, Reason: "dinner",
	})

	// Transfer 150 from Bob to Carol (B: 550, C: 350).
	_ = handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-B", ToAccountID: "acc-C", Amount: 150,
	})

	// Carol deposits 50 (C: 400).
	_ = handler.HandleDeposit(DepositCommand{
		AccountID: "acc-C", Amount: 50, Reason: "refund",
	})

	// Verify the final balances.
	expected := map[string]float64{
		"acc-A": 900.0,
		"acc-B": 550.0,
		"acc-C": 400.0,
	}

	for id, want := range expected {
		got := accountBalance(t, store, id)
		if got != want {
			t.Errorf("%s: expected %.2f, got %.2f", id, want, got)
		}
	}

	// Conservation of money: total should be 1000+500+200 + 200+50
	// (external deposits) - 100 (external withdrawal) = 1850.
	totalBalance := accountBalance(t, store, "acc-A") +
		accountBalance(t, store, "acc-B") +
		accountBalance(t, store, "acc-C")
	if totalBalance != 1850.0 {
		t.Errorf("total across all accounts should be 1850.00 (conservation of money), got %.2f",
			totalBalance)
	}
}
