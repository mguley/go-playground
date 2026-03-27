package account

import (
	"fmt"

	"event-sourcing/pkg/es"
)

// AggregateType is the type identifier used in event metadata.
const AggregateType = "Account"

// Account is the aggregate root for the bank account domain.
// Its state is derived entirely from replaying events - you never
// set these fields directly. The only way to change the account's
// state is to produce an event and apply it.
//
// This separation between "what happened" (events) and "what is the
// current state" (aggregate fields) is the core of event sourcing.
// The events are the source of truth; the fields are a derived cache.
type Account struct {
	// id is the unique identifier for this account.
	id string

	// owner is the name of the account holder.
	owner string

	// balance is the current balance, derived from replaying all
	// deposit and withdrawal events.
	balance float64

	// isOpen tracks whether the account has been opened.
	// An account that has not been opened cannot accept any operations.
	isOpen bool

	// version tracks how many events have been applied. This is used
	// for optimistic concurrency control when saving new events.
	version int

	// changes accumulates uncommitted events produced by command
	// handlers. These are the events that need to be persisted.
	changes []es.Event
}

// NewAccount creates a blank aggregate. It has no state until events
// are applied via LoadFromHistory or by handling commands.
func NewAccount(id string) *Account {
	return &Account{id: id}
}

// ID returns the account's unique identifier.
func (a *Account) ID() string { return a.id }

// Owner returns the account holder's name.
func (a *Account) Owner() string { return a.owner }

// Balance returns the current balance.
func (a *Account) Balance() float64 { return a.balance }

// IsOpen returns whether the account has been opened.
func (a *Account) IsOpen() bool { return a.isOpen }

// Version returns the number of events that have been applied.
func (a *Account) Version() int { return a.version }

// Changes returns the uncommitted events produced by command handlers.
// After persisting these events, call ClearChanges.
func (a *Account) Changes() []es.Event { return a.changes }

// ClearChanges removes all uncommitted events. Called after the
// events have been successfully persisted to the event store.
func (a *Account) ClearChanges() { a.changes = nil }

// LoadFromHistory rebuilds the aggregate's state by replaying a
// sequence of historical events. This is called when loading an
// aggregate from the event store.
//
// The pattern is: create a blank aggregate, load its events from
// the store, and call LoadFromHistory to "fast-forward" the
// aggregate to its current state.
func (a *Account) LoadFromHistory(events []es.Event) {
	for _, event := range events {
		a.apply(event, false)
	}
}

// apply is the core state-transition function. It examines the event
// type and updates the aggregate's fields accordingly.
//
// The isNew parameter distinguishes between replaying historical events
// (isNew=false, do not add to changes) and applying events produced
// by a command handler (isNew=true, add to uncommitted changes).
func (a *Account) apply(event es.Event, isNew bool) {
	switch e := event.Data.(type) {
	case AccountOpenedEvent:
		a.owner = e.Owner
		a.balance = e.InitialBalance
		a.isOpen = true

	case MoneyDepositedEvent:
		a.balance += e.Amount

	case MoneyWithdrawnEvent:
		a.balance -= e.Amount
	}

	a.version++

	if isNew {
		a.changes = append(a.changes, event)
	}
}

// raiseEvent creates a new event and applies it to the aggregate.
// This is the only way command handlers should produce state changes.
func (a *Account) raiseEvent(eventType string, data any) {
	event := es.Event{
		AggregateID:   a.id,
		AggregateType: AggregateType,
		EventType:     eventType,
		Data:          data,
	}
	a.apply(event, true)
}

// --- Command handlers ---
// Each command handler validates business rules against the current
// state and, if valid, raises one or more events. The validation
// happens against the in-memory state (which was rebuilt from events),
// and the result is new events - not direct state mutations.

// OpenAccount initializes the account with an owner and optional
// initial balance. This command can only be executed once per account.
func (a *Account) OpenAccount(owner string, initialBalance float64) error {
	if a.isOpen {
		return fmt.Errorf("account %s is already open", a.id)
	}
	if owner == "" {
		return fmt.Errorf("owner name is required")
	}
	if initialBalance < 0 {
		return fmt.Errorf("initial balance cannot be negative: %.2f", initialBalance)
	}

	a.raiseEvent(EventAccountOpened, AccountOpenedEvent{
		Owner:          owner,
		InitialBalance: initialBalance,
	})
	return nil
}

// Deposit adds funds to the account.
func (a *Account) Deposit(amount float64, reason string) error {
	if !a.isOpen {
		return fmt.Errorf("account %s is not open", a.id)
	}
	if amount <= 0 {
		return fmt.Errorf("deposit amount must be positive: %.2f", amount)
	}
	if reason == "" {
		reason = "deposit"
	}

	a.raiseEvent(EventMoneyDeposited, MoneyDepositedEvent{
		Amount: amount,
		Reason: reason,
	})
	return nil
}

// Withdraw removes funds from the account, subject to a sufficient
// balance check. This is the business rule that prevents overdrafts.
func (a *Account) Withdraw(amount float64, reason string) error {
	if !a.isOpen {
		return fmt.Errorf("account %s is not open", a.id)
	}
	if amount <= 0 {
		return fmt.Errorf("withdrawal amount must be positive: %.2f", amount)
	}
	if a.balance < amount {
		return fmt.Errorf(
			"insufficient balance: have %.2f, need %.2f",
			a.balance, amount,
		)
	}
	if reason == "" {
		reason = "withdrawal"
	}

	a.raiseEvent(EventMoneyWithdrawn, MoneyWithdrawnEvent{
		Amount: amount,
		Reason: reason,
	})
	return nil
}
