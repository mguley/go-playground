package projection

import (
	"testing"

	"event-sourcing/pkg/account"
	"event-sourcing/pkg/es"
)

// TestLedgerProjection_AccountOpenedWithInitialBalance verifies that
// opening an account with a positive initial balance creates an
// initial deposit transaction in the ledger.
func TestLedgerProjection_AccountOpenedWithInitialBalance(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 500}, 0)

	transaction := projection.GetTransactions("acc-001")
	if len(transaction) != 1 {
		t.Fatalf("expected 1 transaction for initial balance, got %d", len(transaction))
	}

	txn := transaction[0]
	if txn.Type != "deposit" {
		t.Errorf("expected type 'deposit', got %q", txn.Type)
	}
	if txn.Amount != 500 {
		t.Errorf("expected amount 500, got %.2f", txn.Amount)
	}
	if txn.Reason != "initial_balance" {
		t.Errorf("expected reason 'initial_balance', got %q", txn.Reason)
	}
	if txn.Balance != 500 {
		t.Errorf("expected running balance 500, got %.2f", txn.Balance)
	}
	if txn.AccountID != "acc-001" {
		t.Errorf("expected account ID 'acc-001', got %q", txn.AccountID)
	}
}

// TestLedgerProjection_AccountOpenedWithZeroBalance verifies that
// opening an account with a zero balance produces no transaction
// entry in the ledger. There is nothing to record.
func TestLedgerProjection_AccountOpenedWithZeroBalance(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 0}, 0)

	transaction := projection.GetTransactions("acc-001")
	// The ledger should have no entries because there is no money
	// movement to record for a zero balance opening.
	if transaction != nil {
		t.Errorf("expected nil transactions for zero-balance open, got %d entries", len(transaction))
	}
}

// TestLedgerProjection_DepositCreatesTransaction verifies that a
// deposit event produces a transaction entry with the correct
// running balance.
func TestLedgerProjection_DepositCreatesTransaction(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 250, Reason: "paycheck"}, 1)

	transaction := projection.GetTransactions("acc-001")
	// Two transactions: the initial_balance deposit and the paycheck.
	if len(transaction) != 2 {
		t.Fatalf("expected 2 transactions, got %d", len(transaction))
	}

	deposit := transaction[1]
	if deposit.Type != "deposit" {
		t.Errorf("expected type 'deposit', got %q", deposit.Type)
	}
	if deposit.Amount != 250 {
		t.Errorf("expected amount 250, got %.2f", deposit.Amount)
	}
	if deposit.Reason != "paycheck" {
		t.Errorf("expected reason 'paycheck', got %q", deposit.Reason)
	}
	if deposit.Balance != 350 {
		t.Errorf("expected running balance 350 (100+250), got %.2f", deposit.Balance)
	}
}

// TestLedgerProjection_WithdrawalCreatesTransaction verifies that a
// withdrawal event produces a transaction with type "withdrawal"
// and the correct running balance.
func TestLedgerProjection_WithdrawalCreatesTransaction(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 500}, 0)
	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 150, Reason: "groceries"}, 1)

	transaction := projection.GetTransactions("acc-001")
	if len(transaction) != 2 {
		t.Fatalf("expected 2 transactions, got %d", len(transaction))
	}

	withdrawal := transaction[1]
	if withdrawal.Type != "withdrawal" {
		t.Errorf("expected type 'withdrawal', got %q", withdrawal.Type)
	}
	if withdrawal.Amount != 150 {
		t.Errorf("expected amount 150, got %.2f", withdrawal.Amount)
	}
	if withdrawal.Reason != "groceries" {
		t.Errorf("expected reason 'groceries', got %q", withdrawal.Reason)
	}
	if withdrawal.Balance != 350 {
		t.Errorf("expected running balance 350 (500-150), got %.2f", withdrawal.Balance)
	}
}

// TestLedgerProjection_RunningBalanceAcrossMultipleOperations
// verifies that the running balance column in the ledger correctly
// reflects every operation in sequence: open, deposit, withdraw,
// deposit, withdraw.
func TestLedgerProjection_RunningBalanceAcrossMultipleOperations(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	// Open with $200, +$300, -$100, +$50, -$25.
	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 200}, 0)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 300, Reason: "bonus"}, 1)
	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 100, Reason: "rent"}, 2)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 50, Reason: "refund"}, 3)
	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 25, Reason: "coffee"}, 4)

	transaction := projection.GetTransactions("acc-001")
	if len(transaction) != 5 {
		t.Fatalf("expected 5 transactions, got %d", len(transaction))
	}

	// Verify the running balance after each entry.
	expectedBalances := []float64{200, 500, 400, 450, 425}
	for i, expected := range expectedBalances {
		if transaction[i].Balance != expected {
			t.Errorf("transaction %d: expected running balance %.2f, got %.2f",
				i, expected, transaction[i].Balance)
		}
	}
}

// TestLedgerProjection_GetRecentTransactions verifies that the
// "last N" query returns the correct tail of the transaction list.
func TestLedgerProjection_GetRecentTransactions(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 200, Reason: "deposit-1"}, 1)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 300, Reason: "deposit-2"}, 2)
	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 50, Reason: "withdrawal-1"}, 3)

	// There are 4 total transactions (initial + 3 operations).
	// Ask for the last 2.
	recent := projection.GetRecentTransactions("acc-001", 2)
	if len(recent) != 2 {
		t.Fatalf("expected 2 recent transactions, got %d", len(recent))
	}

	// The last two should be deposit-2 and withdrawal-1.
	if recent[0].Reason != "deposit-2" {
		t.Errorf("expected first recent txn reason 'deposit-2', got %q", recent[0].Reason)
	}
	if recent[1].Reason != "withdrawal-1" {
		t.Errorf("expected second recent txn reason 'withdrawal-1', got %q", recent[1].Reason)
	}
}

// TestLedgerProjection_GetRecentTransactionsExceedsTotal verifies
// that requesting more transactions than exist returns all of them.
func TestLedgerProjection_GetRecentTransactionsExceedsTotal(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 50, Reason: "tip"}, 1)

	// There are 2 transactions; ask for 10.
	recent := projection.GetRecentTransactions("acc-001", 10)
	if len(recent) != 2 {
		t.Errorf("expected 2 transactions (all of them), got %d", len(recent))
	}
}

// TestLedgerProjection_GetTransactionsUnknownAccount verifies that
// querying transactions for a nonexistent account returns nil.
func TestLedgerProjection_GetTransactionsUnknownAccount(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	transaction := projection.GetTransactions("no-such-account")
	if transaction != nil {
		t.Errorf("expected nil for unknown account, got %d entries", len(transaction))
	}
}

// TestLedgerProjection_TransactionsReturnCopies verifies that the
// returned slice is a copy: mutating it does not corrupt the
// projection's internal state.
func TestLedgerProjection_TransactionsReturnCopies(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)

	// Get transactions and mutate the returned slice.
	transaction := projection.GetTransactions("acc-001")
	transaction[0].Amount = 999_999

	// Fetch again - internal state should be unaffected.
	fresh := projection.GetTransactions("acc-001")
	if fresh[0].Amount != 100 {
		t.Errorf("expected internal amount to remain 100, got %.2f", fresh[0].Amount)
	}
}

// TestLedgerProjection_MultipleAccountsIndependent verifies that
// transactions for different accounts are tracked separately.
func TestLedgerProjection_MultipleAccountsIndependent(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 1000}, 0)
	saveEvent(t, store, "acc-002", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Bob", InitialBalance: 500}, 0)

	// Deposit only into Alice's account.
	saveEvent(t, store, "acc-001", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 200, Reason: "bonus"}, 1)

	aliceTransaction := projection.GetTransactions("acc-001")
	bobTransaction := projection.GetTransactions("acc-002")

	if len(aliceTransaction) != 2 {
		t.Errorf("alice: expected 2 transactions, got %d", len(aliceTransaction))
	}
	if len(bobTransaction) != 1 {
		t.Errorf("bob: expected 1 transaction, got %d", len(bobTransaction))
	}

	// Verify Bob's balance was not affected.
	if bobTransaction[0].Balance != 500 {
		t.Errorf("bob: expected balance 500, got %.2f", bobTransaction[0].Balance)
	}
}

// TestLedgerProjection_TransferProducesCorrectEntries simulates a
// transfer by emitting a withdrawal on one account and a deposit on
// another, then verifies that both ledgers reflect the movement with
// the correct reasons and running balances.
func TestLedgerProjection_TransferProducesCorrectEntries(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 1000}, 0)
	saveEvent(t, store, "acc-002", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Bob", InitialBalance: 200}, 0)

	// Simulate a $300 transfer from Alice to Bob.
	saveEvent(t, store, "acc-001", account.EventMoneyWithdrawn,
		account.MoneyWithdrawnEvent{Amount: 300, Reason: "transfer_out"}, 1)
	saveEvent(t, store, "acc-002", account.EventMoneyDeposited,
		account.MoneyDepositedEvent{Amount: 300, Reason: "transfer_in"}, 1)

	aliceTransaction := projection.GetTransactions("acc-001")
	bobTransaction := projection.GetTransactions("acc-002")

	// Alice: initial_balance deposit + transfer_out withdrawal.
	if len(aliceTransaction) != 2 {
		t.Fatalf("alice: expected 2 transactions, got %d", len(aliceTransaction))
	}
	transferOut := aliceTransaction[1]
	if transferOut.Type != "withdrawal" {
		t.Errorf("alice transfer: expected type 'withdrawal', got %q", transferOut.Type)
	}
	if transferOut.Reason != "transfer_out" {
		t.Errorf("alice transfer: expected reason 'transfer_out', got %q", transferOut.Reason)
	}
	if transferOut.Balance != 700 {
		t.Errorf("alice transfer: expected balance 700, got %.2f", transferOut.Balance)
	}

	// Bob: initial_balance deposit + transfer_in deposit.
	if len(bobTransaction) != 2 {
		t.Fatalf("bob: expected 2 transactions, got %d", len(bobTransaction))
	}
	transferIn := bobTransaction[1]
	if transferIn.Type != "deposit" {
		t.Errorf("bob transfer: expected type 'deposit', got %q", transferIn.Type)
	}
	if transferIn.Reason != "transfer_in" {
		t.Errorf("bob transfer: expected reason 'transfer_in', got %q", transferIn.Reason)
	}
	if transferIn.Balance != 500 {
		t.Errorf("bob transfer: expected balance 500, got %.2f", transferIn.Balance)
	}
}

// TestLedgerProjection_TransactionHasEventID verifies that each
// transaction record captures the event ID assigned by the store,
// which links the ledger entry back to the source event for
// traceability.
func TestLedgerProjection_TransactionHasEventID(t *testing.T) {
	store := es.NewInMemoryEventStore()
	projection := NewLedgerProjection(store)

	saveEvent(t, store, "acc-001", account.EventAccountOpened,
		account.AccountOpenedEvent{Owner: "Alice", InitialBalance: 100}, 0)

	transaction := projection.GetTransactions("acc-001")
	if len(transaction) != 1 {
		t.Fatalf("expected 1 transaction, got %d", len(transaction))
	}
	if transaction[0].EventID == "" {
		t.Error("expected transaction to have a non-empty EventID")
	}
}
