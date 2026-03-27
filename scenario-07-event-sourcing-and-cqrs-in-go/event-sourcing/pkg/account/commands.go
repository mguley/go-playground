package account

import (
	"fmt"

	"event-sourcing/pkg/es"
)

// Commands represent intentions to change the system. Unlike events
// (which describe what happened), commands describe what someone
// wants to happen. A command may be rejected if business rules
// are violated.

// OpenAccountCommand requests the creation of a new account.
type OpenAccountCommand struct {
	AccountID      string
	Owner          string
	InitialBalance float64
}

// DepositCommand requests a deposit into an existing account.
type DepositCommand struct {
	AccountID string
	Amount    float64
	Reason    string
}

// WithdrawCommand requests a withdrawal from an existing account.
type WithdrawCommand struct {
	AccountID string
	Amount    float64
	Reason    string
}

// TransferCommand requests a transfer between two accounts.
// This is the most complex command because it spans two aggregates.
type TransferCommand struct {
	FromAccountID string
	ToAccountID   string
	Amount        float64
}

// CommandHandler processes commands by coordinating between the
// event store and domain aggregates. It implements the standard
// event-sourcing command processing pipeline:
//
//  1. Load the aggregate's event history from the store
//  2. Rebuild the aggregate's state by replaying those events
//  3. Execute the command against the aggregate (validate + raise events)
//  4. Persist the new events to the store with optimistic concurrency
//
// This pipeline guarantees that business rules are evaluated against
// the most recent state and that concurrent modifications are detected.
type CommandHandler struct {
	store es.EventStore
}

// NewCommandHandler creates a command handler backed by the given
// event store.
func NewCommandHandler(store es.EventStore) *CommandHandler {
	return &CommandHandler{store: store}
}

// HandleOpenAccount processes an account opening command.
func (h *CommandHandler) HandleOpenAccount(cmd OpenAccountCommand) error {
	if cmd.AccountID == "" {
		return fmt.Errorf("account ID is required")
	}

	// Load any existing events for this aggregate.
	events, err := h.store.LoadEvents(cmd.AccountID)
	if err != nil {
		return fmt.Errorf("loading events: %w", err)
	}

	// If events already exist, the account has already been opened.
	// We still rebuild and let the aggregate's own validation catch
	// this, which provides a clearer error message.
	account := NewAccount(cmd.AccountID)
	account.LoadFromHistory(events)

	// Execute the command. The aggregate validates business rules
	// and raises events if valid.
	if err = account.OpenAccount(cmd.Owner, cmd.InitialBalance); err != nil {
		return err
	}

	// Persist the new events with optimistic concurrency control.
	return h.store.SaveEvents(
		cmd.AccountID,
		account.Changes(),
		account.Version()-len(account.Changes()),
	)
}

// HandleDeposit processes a deposit command.
func (h *CommandHandler) HandleDeposit(cmd DepositCommand) error {
	account, err := h.loadAccount(cmd.AccountID)
	if err != nil {
		return err
	}

	if err = account.Deposit(cmd.Amount, cmd.Reason); err != nil {
		return err
	}

	return h.saveChanges(account)
}

// HandleWithdraw processes a withdrawal command.
func (h *CommandHandler) HandleWithdraw(cmd WithdrawCommand) error {
	account, err := h.loadAccount(cmd.AccountID)
	if err != nil {
		return err
	}

	if err = account.Withdraw(cmd.Amount, cmd.Reason); err != nil {
		return err
	}

	return h.saveChanges(account)
}

// HandleTransfer processes a transfer between two accounts.
//
// Transfers are inherently complex in event sourcing because they
// span two aggregates. We handle this by processing both sides
// sequentially: withdraw from the source, then deposit to the
// destination. If the deposit fails (which should not happen for
// valid accounts, but defensive coding demands we handle it), the
// system is left in an inconsistent state that would need a
// compensating action to resolve.
//
// A production system would use the Saga pattern or a process
// manager to handle this more robustly. The Saga would define
// the transfer as a multistep workflow with explicit compensation:
// if the deposit step fails, a compensating "refund" event is
// emitted on the source account to reverse the withdrawal.
func (h *CommandHandler) HandleTransfer(cmd TransferCommand) error {
	if cmd.FromAccountID == cmd.ToAccountID {
		return fmt.Errorf("cannot transfer to the same account")
	}
	if cmd.Amount <= 0 {
		return fmt.Errorf("transfer amount must be positive: %.2f", cmd.Amount)
	}

	// Step 1: Withdraw from the source account.
	fromAccount, err := h.loadAccount(cmd.FromAccountID)
	if err != nil {
		return fmt.Errorf("loading source account: %w", err)
	}

	if err = fromAccount.Withdraw(cmd.Amount, "transfer_out"); err != nil {
		return fmt.Errorf("source withdrawal: %w", err)
	}

	if err = h.saveChanges(fromAccount); err != nil {
		return fmt.Errorf("saving source withdrawal: %w", err)
	}

	// Step 2: Deposit to the destination account.
	toAccount, err := h.loadAccount(cmd.ToAccountID)
	if err != nil {
		return fmt.Errorf("loading destination account: %w", err)
	}

	if err = toAccount.Deposit(cmd.Amount, "transfer_in"); err != nil {
		return fmt.Errorf("destination deposit: %w", err)
	}

	if err = h.saveChanges(toAccount); err != nil {
		return fmt.Errorf("saving destination deposit: %w", err)
	}

	return nil
}

// loadAccount loads an aggregate from the event store and rebuilds its state.
func (h *CommandHandler) loadAccount(id string) (*Account, error) {
	events, err := h.store.LoadEvents(id)
	if err != nil {
		return nil, fmt.Errorf("loading events for %s: %w", id, err)
	}

	account := NewAccount(id)
	account.LoadFromHistory(events)
	return account, nil
}

// saveChanges persists the aggregate's uncommitted events to the
// store. The expected version is calculated from the aggregate's
// current version minus the number of uncommitted changes.
func (h *CommandHandler) saveChanges(account *Account) error {
	expectedVersion := account.Version() - len(account.Changes())
	return h.store.SaveEvents(
		account.ID(),
		account.Changes(),
		expectedVersion,
	)
}
