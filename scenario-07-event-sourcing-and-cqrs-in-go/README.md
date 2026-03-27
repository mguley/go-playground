# Event sourcing and CQRS in Go

## Table of Contents
- [Introduction](#introduction)
- [What is event sourcing?](#what-is-event-sourcing)
- [What is CQRS?](#what-is-cqrs)
- [Why combine event sourcing with CQRS?](#why-combine-event-sourcing-with-cqrs)
- [Prerequisites](#prerequisites)
- [Step 1: Understanding our domain - a bank account system](#step-1-understanding-our-domain---a-bank-account-system)
- [Step 2: Defining events - the source of truth](#step-2-defining-events---the-source-of-truth)
- [Step 3: Building the event store](#step-3-building-the-event-store)
- [Step 4: Building the aggregate - Account](#step-4-building-the-aggregate---account)
- [Step 5: Command handling - enforcing business rules](#step-5-command-handling---enforcing-business-rules)
- [Step 6: Building read models - projections](#step-6-building-read-models---projections)
- [Step 7: Snapshotting - taming long event streams](#step-7-snapshotting---taming-long-event-streams)
- [Step 8: Testing event-sourced systems](#step-8-testing-event-sourced-systems)
- [Step 9: Putting it all together](#step-9-putting-it-all-together)
- [Conclusion](#conclusion)

---

#### Introduction

Picture this: your team maintains a payment system. A customer reports that `$200` vanished from their account.
You check the database and see the current balance is `$300`. What happened? Was it a legitimate withdrawal? A failed transfer?
A double-charge from a retry? A defect in the balance calculation?

The database holds one number: 300. The history that produced that number is gone.
Your team spends hours digging through application logs, correlating timestamps, and interviewing the customer about what operations they performed.
Eventually, an engineer traces the issue to a race condition where two concurrent withdrawals both read the same balance, both passed the "sufficient funds" check, and both decremented the same snapshot.
The database dutifully recorded the final state after each write, but the intermediate state - the fact that two operations overlapped, left no trace in the data model itself.

This is the fundamental limitation of state-based persistence: **you store what things are, not how they got there.**

Event sourcing flips this model. Instead of storing the current balance, you store every event that ever affected the account: `AccountOpened`, `MoneyDeposited`, `MoneyWithdrawn`, `TransferSent`.
The current balance is not stored at all - it is computed by replaying these events from the beginning.
If a customer says `"$200 vanished"` you do not dig through logs. You read the event stream and see exactly what happened, in what order, with full context.

CQRS (Command Query Responsibility Segregation) is the natural companion to event sourcing.
When your source of truth is a stream of events, reading the current state requires replaying all events - which is expensive for queries.
CQRS solves this by separating the write side (which appends events) from the read side (which maintains pre-computed projections optimized for queries).
- The write side enforces business rules and produces events.
- The read side subscribes to those events and builds whatever views the application needs: account balances, transaction histories, audit logs, analytics dashboards.

In this deep dive, we will build a complete bank account system from scratch using event sourcing and CQRS in Go.
You will implement an event store, aggregates that enforce business rules, command handlers that coordinate operations, read-model projections that answer queries efficiently,
and snapshots that prevent event replay from becoming a bottleneck.
By the end, you will understand not just the patterns but the practical tradeoffs and pitfalls that determine whether event sourcing helps or hurts a real system.

---

#### What is event sourcing?

In a traditional system, you store the current state of an entity. A bank account row might look like:

```
| account_id | owner  | balance | updated_at          |
|------------|--------|---------|---------------------|
| acc-001    | Alice  | 750.00  | 2026-03-15 14:22:00 |
```

When Alice deposits `$200`, you run `UPDATE accounts SET balance = 950.00 WHERE account_id = 'acc-001'`. The previous balance of `$750` is gone.
If you want history, you must bolt on an audit log or change data capture pipeline as an afterthought.

Event sourcing inverts this. Instead of storing state, you store the sequence of events that produced that state:

```
| event_id | aggregate_id | type            | data                                     | version |
|----------|--------------|-----------------|------------------------------------------|---------|
| evt-001  | acc-001      | AccountOpened   | {"owner": "Alice", "initial_balance": 0} | 1       |
| evt-002  | acc-001      | MoneyDeposited  | {"amount": 500}                          | 2       |
| evt-003  | acc-001      | MoneyDeposited  | {"amount": 250}                          | 3       |
| evt-004  | acc-001      | MoneyDeposited  | {"amount": 200}                          | 4       |
```

To know Alice's current balance, you replay these events: start at 0 (opened), add 500, add 250, add 200. The balance is 950.
The event stream is an append-only log. Events are immutable facts that have already occurred - you never update or delete them.

**What event sourcing gives you**

- Complete audit trail: every state change is recorded as an event with full context.
  You do not need a separate audit log because the event stream *is* the audit log.

- Temporal queries: you can reconstruct the state of any entity at any point in time by replaying events up to that moment.
  "What was Alice's balance on March 10th?" is a trivial query in an event-sourced system.

- Debugging and root cause analysis: when something goes wrong, you have the complete causal history.
  The race condition from our opening example would be immediately visible as two concurrent events in the stream.

- Event-driven integration: other systems can subscribe to your event stream and react to changes without polling.
  Your notification service, analytics pipeline, and fraud detection system all consume the same events.

**What event sourcing costs you**

- Eventual consistency: read models are updated asynchronously, so queries may return slightly stale data.
  Systems that require "read your own writes" semantics need careful design.

- Complexity: the programming model is less intuitive than CRUD.
  Developers must think in terms of events, aggregates, and projections rather than simple database rows.

- Event schema evolution: once an event is stored, its structure becomes a contract.
  Changing that structure without breaking old events requires versioning strategies (upcasting, event migration, or tolerant readers).

- Storage growth: event streams grow without bound.
  Snapshotting and archival strategies are necessary for long-lived aggregates.

---

#### What is CQRS?

CQRS stands for Command Query Responsibility Segregation. The core idea is to separate the model that handles writes (commands) from the model that handles reads (queries).

In a traditional application, the same data model handles both.
Your `Account` struct has fields for the balance, and the same struct is used to process deposits (writes) and to return account details to the UI (reads). This seems natural, but it creates tension.
The write model needs to enforce business invariants (`"you cannot withdraw more than your balance"`), while the read model needs to answer diverse queries efficiently (`"show me all accounts with balances above $10,000 sorted by last activity"`).
Optimizing for one often compromises the other.

CQRS resolves this tension by splitting the responsibilities:

- The **command side** (write model) processes commands like `DepositMoney` and `WithdrawMoney`.
  It loads the aggregate from the event store, applies business rules, and emits new events. It is optimized for consistency and invariant enforcement.

- The **query side** (read model) maintains one or more projections that are updated by subscribing to the event stream. Each projection is optimized for a specific query pattern.
  An `"account balance"` projection might be a simple key-value lookup, while a `"transaction history"` projection might be a time-ordered list with filtering and pagination support.

The two sides communicate through events. The command side produces events, and the query side consumes them.
This separation means you can scale reads and writes independently, use different storage technologies for each side (an event store for writes, a search index for reads),
and add new query patterns without touching the write model.

---

#### Why combine event sourcing with CQRS?

Event sourcing and CQRS are independent patterns, but they complement each other so well that they are almost always used together.

Event sourcing needs CQRS because replaying an entire event stream to answer a query is expensive.
If Alice's account has 10,000 events, computing her balance requires replaying all 10,000 events every time someone checks the balance.
Pre-computed read models solve this problem by maintaining the current balance as a materialized view that is updated incrementally as new events arrive.

CQRS benefits from event sourcing because the event stream provides a reliable mechanism for keeping read models in sync with the write model.
Without event sourcing, the write side must explicitly notify the read side of changes, which creates coupling.
With event sourcing, the read side simply subscribes to the event stream - the same stream that is the write model's source of truth.

Together, they form a cohesive architecture where the event store is the single source of truth, commands validate business rules and append events,
and projections consume events to build query-optimized views.

---

#### Prerequisites

Before we begin, ensure you have the following:

- Go 1.26 or later installed
- A code editor of your choice
- Basic understanding of Go interfaces, structs, and error handling
- Familiarity with Go testing and the `testing` package
- A terminal for running commands

---

#### Step 1: Understanding our domain - a bank account system

Our domain is a simplified bank account system. It is complex enough to demonstrate every aspect of event sourcing and CQRS, but simple enough to build in a single walkthrough.

**The operations we support**

- Opening an account: creates a new account with an owner name and an initial deposit (which can be zero).

- Depositing money: adds funds to an existing account.

- Withdrawing money: removes funds from an existing account, subject to a "sufficient balance" business rule.

- Transferring money: moves funds from one account to another.
  This is the most interesting operation because it spans two aggregates and demonstrates how event sourcing handles cross-aggregate coordination.

**The events our system produces**

Each operation, if successful, appends one or more events to the event store:

- `AccountOpened` - an account was created with an owner and optional initial balance.

- `MoneyDeposited` - funds were added to an account, with the amount and a reason (deposit, transfer-in, interest, etc.).

- `MoneyWithdrawn` - funds were removed from an account, with the amount and a reason (withdrawal, transfer-out, fee, etc.).

**The read models we will build**

- Account balance view: given an account ID, returns the current balance and owner. This is the simplest projection and demonstrates the basic pattern.

- Transaction ledger view: given an account ID, returns the ordered list of all transactions (deposits and withdrawals) with timestamps, amounts, and reasons.
  This demonstrates a list-based projection.

**Why this is a good event sourcing domain**

Bank accounts are the canonical example for event sourcing because the domain naturally thinks in terms of transactions (events),
the current balance is a derived value (computed from transaction history), audit requirements demand complete traceability,
and concurrency matters (two withdrawals should not both succeed if the balance is insufficient for both).

**A note on monetary types:** throughout this tutorial, we use `float64` for monetary amounts to keep the focus on event sourcing patterns rather than numeric precision.
In a production financial system, you should use an integer type representing the smallest currency unit (e.g., cents) or a dedicated decimal library like `shopspring/decimal`.
Floating-point arithmetic introduces rounding errors that are unacceptable in real financial software.

Let us create our project structure and initialize the module:

```bash
mkdir -p event-sourcing/{pkg/es,pkg/account,pkg/projection}
cd event-sourcing
go mod init event-sourcing
```

Your `go.mod` file should look like this:

```
module event-sourcing

go 1.26.0
```

---

#### Step 2: Defining events - the source of truth

Events are the foundation of an event-sourced system. Every event is an immutable record of something that happened.
Note the past tense: events describe facts, not intentions. `MoneyDeposited` (past tense, fact) rather than `DepositMoney` (imperative, command).

Let us start with the core event infrastructure, then define the domain-specific events.

Create `pkg/es/event.go`:

```go
package es

import (
	"time"
)

// Event represents a single domain event in the system. Events are
// immutable records of something that happened. They are the source
// of truth in an event-sourced system - the current state of any
// entity is derived by replaying its events from the beginning.
//
// The Version field is crucial for optimistic concurrency control:
// when saving new events, the store checks that the expected version
// matches the current version of the aggregate, preventing lost-update
// problems from concurrent command processing.
type Event struct {
	// ID uniquely identifies this event. Assigned by the event store
	// when the event is persisted.
	ID string

	// AggregateID identifies which aggregate (entity) this event
	// belongs to. All events for a single bank account share the
	// same AggregateID.
	AggregateID string

	// AggregateType is the type name of the aggregate (e.g., "Account").
	// This allows a single event store to host multiple aggregate types.
	AggregateType string

	// EventType is the name of the event (e.g., "AccountOpened",
	// "MoneyDeposited"). Used for deserialization and routing.
	EventType string

	// Version is the sequence number of this event within its aggregate
	// stream. The first event for an aggregate has version 1, the second
	// has version 2, and so on. Versions must be contiguous - gaps
	// indicate data corruption or a defect.
	Version int

	// Timestamp records when the event occurred.
	Timestamp time.Time

	// Data holds the event-specific payload. The concrete type depends
	// on EventType - for example, a "MoneyDeposited" event's Data field
	// holds a MoneyDepositedEvent struct.
	Data any
}
```

Now define the domain events for our bank account system.

Create `pkg/account/events.go`:

```go
package account

// AccountOpenedEvent is recorded when a new bank account is created.
// It captures the owner's name and the initial deposit amount.
// Every account's event stream starts with exactly one of these.
type AccountOpenedEvent struct {
	Owner          string  `json:"owner"`
	InitialBalance float64 `json:"initial_balance"`
}

// MoneyDepositedEvent is recorded when funds are added to an account.
// The Reason field provides context: "deposit" for a direct deposit,
// "transfer_in" for funds received from another account, "interest"
// for accrued interest, and so on.
type MoneyDepositedEvent struct {
	Amount float64 `json:"amount"`
	Reason string  `json:"reason"`
}

// MoneyWithdrawnEvent is recorded when funds are removed from an account.
// The Reason field provides context: "withdrawal" for a direct withdrawal,
// "transfer_out" for funds sent to another account, "fee" for charges, etc.
type MoneyWithdrawnEvent struct {
	Amount float64 `json:"amount"`
	Reason string  `json:"reason"`
}

// Event type constants. These are used as the EventType field in the
// generic Event envelope and for routing events to the correct handler.
const (
	EventAccountOpened  = "AccountOpened"
	EventMoneyDeposited = "MoneyDeposited"
	EventMoneyWithdrawn = "MoneyWithdrawn"
)
```

A few design decisions are worth noting here.

Events carry only the data that changed, not the full state. `MoneyDepositedEvent` stores the deposit amount, not the resulting balance.
The balance is derived by replaying all events; storing it in the event would create redundancy that could become inconsistent.

Events use string types for identification and routing rather than Go's type system directly.
This is deliberate: events are serialized for storage and may be consumed by systems written in different languages. The `EventType` string is the contract, not the Go struct type.

The `Reason` field on deposit and withdrawal events is a simple form of context that makes the event stream self-documenting.
When you read the event stream, you can distinguish a customer withdrawal from a transfer debit without cross-referencing other data.

---

#### Step 3: Building the event store

The event store is the persistence layer for events.

It is an append-only log that supports two primary operations:
- appending new events for an aggregate (with optimistic concurrency)
- and loading all events for an aggregate (to rebuild its state)

Our implementation is in-memory, which keeps the focus on the patterns rather than database mechanics.
A production event store would use a database like PostgreSQL (with an events table), EventStoreDB, or a message log like Apache Kafka, but the interface would remain the same.

Create `pkg/es/store.go`:

```go
package es

import (
	"fmt"
	"sync"
	"time"

	"crypto/rand"
	"encoding/hex"
)

// EventStore defines the interface for persisting and retrieving events.
// Any implementation (in-memory, PostgreSQL, EventStoreDB) must satisfy
// this contract.
type EventStore interface {
	// SaveEvents appends new events to the aggregate's stream.
	//
	// The expectedVersion parameter enables optimistic concurrency
	// control: if the current version of the aggregate in the store
	// does not match expectedVersion, the save is rejected with a
	// ConcurrencyError. This prevents two concurrent command handlers
	// from silently overwriting each other's changes.
	SaveEvents(aggregateID string, events []Event, expectedVersion int) error

	// LoadEvents retrieves all events for a given aggregate, ordered
	// by version. The returned events can be replayed to reconstruct
	// the aggregate's current state.
	LoadEvents(aggregateID string) ([]Event, error)

	// LoadEventsFrom retrieves events for a given aggregate starting
	// from a specific version (inclusive). This is used in combination
	// with snapshots: load the snapshot (which records the version it
	// was taken at), then load only events after that version.
	LoadEventsFrom(aggregateID string, fromVersion int) ([]Event, error)
}

// ConcurrencyError is returned when a save operation fails because
// another process has already written events past the expected version.
// This is not a defect - it is the optimistic concurrency control mechanism
// doing its job. The caller should reload the aggregate, re-evaluate the command, and retry.
type ConcurrencyError struct {
	AggregateID     string
	ExpectedVersion int
	ActualVersion   int
}

func (e *ConcurrencyError) Error() string {
	return fmt.Sprintf(
		"concurrency conflict on aggregate %s: expected version %d, but current version is %d",
		e.AggregateID, e.ExpectedVersion, e.ActualVersion,
	)
}

// InMemoryEventStore is a thread-safe, in-memory implementation of
// EventStore. It stores events in a map keyed by aggregate ID, with
// each value being an ordered slice of events.
//
// This implementation is suitable for testing, prototyping, and
// learning. A production system would replace it with a durable store
// backed by a database.
type InMemoryEventStore struct {
	mu     sync.RWMutex
	events map[string][]Event // aggregateID -> ordered events

	// subscribers receive every event that is successfully saved.
	// This is the mechanism by which read-model projections stay
	// in sync with the write model.
	subscribers []func(Event)
}

// NewInMemoryEventStore creates a new empty event store.
func NewInMemoryEventStore() *InMemoryEventStore {
	return &InMemoryEventStore{
		events: make(map[string][]Event),
	}
}

// Subscribe registers a callback that will be invoked for every
// event that is successfully persisted. Projections use this to
// update their read models in response to new events.
//
// In a production system, this would be replaced by a durable
// subscription mechanism (e.g., polling the event store, a message
// broker, or a change data capture stream).
func (s *InMemoryEventStore) Subscribe(handler func(Event)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.subscribers = append(s.subscribers, handler)
}

// SaveEvents persists new events with optimistic concurrency control.
// It validates that the expected version matches the current stream
// length, assigns event metadata (ID, timestamp, version), and
// notifies all subscribers.
func (s *InMemoryEventStore) SaveEvents(aggregateID string, events []Event, expectedVersion int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	existing := s.events[aggregateID]
	currentVersion := len(existing)

	// Optimistic concurrency check: the caller must know the current
	// version to append new events. If another goroutine has already
	// appended events since the caller loaded the aggregate, this
	// check fails and the caller must retry.
	if currentVersion != expectedVersion {
		return &ConcurrencyError{
			AggregateID:     aggregateID,
			ExpectedVersion: expectedVersion,
			ActualVersion:   currentVersion,
		}
	}

	// Assign metadata to each event and append to the stream.
	enriched := make([]Event, len(events))
	for i, event := range events {
		event.ID = generateEventID()
		event.Version = currentVersion + i + 1
		event.Timestamp = time.Now()
		enriched[i] = event
	}

	s.events[aggregateID] = append(existing, enriched...)

	// Notify subscribers. In production you would notify outside
	// the critical section, but for simplicity we notify while
	// holding the lock. This guarantees subscribers see events
	// in order.
	for _, event := range enriched {
		for _, sub := range s.subscribers {
			sub(event)
		}
	}

	return nil
}

// LoadEvents returns all events for an aggregate, ordered by version.
func (s *InMemoryEventStore) LoadEvents(aggregateID string) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	events := s.events[aggregateID]
	if len(events) == 0 {
		return nil, nil
	}

	// Return a copy to prevent mutation of the store's internal data.
	result := make([]Event, len(events))
	copy(result, events)
	return result, nil
}

// LoadEventsFrom returns events starting from a specific version
// (inclusive). Used for loading events after a snapshot.
func (s *InMemoryEventStore) LoadEventsFrom(aggregateID string, fromVersion int) ([]Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	allEvents := s.events[aggregateID]
	if len(allEvents) == 0 {
		return nil, nil
	}

	// Events are 1-indexed by version but stored 0-indexed in the slice.
	startIdx := fromVersion - 1
	if startIdx < 0 {
		startIdx = 0
	}
	if startIdx >= len(allEvents) {
		return nil, nil
	}

	result := make([]Event, len(allEvents)-startIdx)
	copy(result, allEvents[startIdx:])
	return result, nil
}

// generateEventID creates a random hex string for event identification.
func generateEventID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
```

The most important concept in this code is **optimistic concurrency control** via the `expectedVersion` parameter.

Consider what happens without it: two goroutines both load an account with balance `$500` (version 5).
Both process a `$400` withdrawal, both see sufficient funds, and both append a `MoneyWithdrawn` event.
The account now has two `$400` withdrawals for a total of `$800` withdrawn, but the balance was only `$500`. Money was created from thin air.

With optimistic concurrency, both goroutines load version 5. The first to save succeeds (expected 5, actual 5, match).
The second attempts to save expecting version 5, but the actual version is now 6 (the first save incremented it).
The `ConcurrencyError` tells the second goroutine to reload, re-evaluate, and it will now see a balance of `$100` - insufficient for a `$400` withdrawal.
The command is correctly rejected.

---

#### Step 4: Building the aggregate - account

An aggregate is the central concept in event sourcing. It is the entity that receives commands, enforces business rules, and produces events.
The key principle is: **the aggregate's state is derived entirely from its event history.** There is no separate database row or cached state that could drift out of sync.

The aggregate has two responsibilities: applying events to update its internal state (the "left fold" over the event stream),
and handling commands by validating business rules against the current state and producing new events.

Create `pkg/account/aggregate.go`:

```go
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
```

There are two patterns in this code that are fundamental to event-sourced aggregates.

- The first is the **separation between validation and state change**.
  The `Withdraw` method checks `a.balance < amount` (validation) and, if valid, calls `raiseEvent` (state change through an event). The aggregate never writes `a.balance -= amount` directly.
  Instead, it raises a `MoneyWithdrawnEvent`, and the `apply` method handles the state change. This discipline ensures that every state change is captured as an event.

- The second is the **dual path through `apply`**. When replaying historical events (`isNew=false`), the method updates state but does not add to `changes`.
  When processing a new command (`isNew=true`), it updates state *and* adds to `changes`.
  This means the aggregate's state is always consistent whether it was rebuilt from history or modified by a command, and the uncommitted changes are cleanly available for persistence.

---

#### Step 5: Command handling - enforcing business rules

The command handler is the coordinator between the event store and the aggregate.
It loads the aggregate from its event history, delegates the command to the aggregate (which validates and raises events), and persists the resulting events back to the store.

Create `pkg/account/commands.go`:

```go
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
```

The `HandleTransfer` method deserves special attention. Transfers span two aggregates, and event sourcing does not support distributed transactions across aggregate boundaries.
We handle this with a simple sequential approach: withdraw first, then deposit.
This works for our tutorial, but a production system would use a Saga or a process manager to handle the case where the deposit step fails after the withdrawal has already been committed.
The Saga would emit a compensating `MoneyDeposited` event (with reason `"transfer_refund"`) on the source account to reverse the withdrawal.

Notice the command handler's pipeline: load events, rebuild aggregate, execute command, save changes. This four step flow is identical for every command.
The aggregate encapsulates the business rules, and the handler provides the infrastructure coordination.
This separation keeps business logic testable without an event store and keeps infrastructure concerns out of domain code.

---

#### Step 6: Building read models - projections

The write side is complete: commands produce events, and events are persisted. Now we build the read side.

A projection (also called a read model or view) subscribes to the event stream and maintains a pre-computed data structure optimized for queries.
Instead of replaying all events every time someone asks for an account balance, the projection updates its internal state incrementally as each new event arrives.

Create `pkg/projection/balance.go`:

```go
package projection

import (
	"sort"
	"sync"

	"event-sourcing/pkg/account"
	"event-sourcing/pkg/es"
)

// AccountView represents the data returned by the balance projection.
// This is a read-optimized structure - it contains exactly the fields
// that balance queries need, nothing more.
type AccountView struct {
	AccountID string  `json:"account_id"`
	Owner     string  `json:"owner"`
	Balance   float64 `json:"balance"`
	Version   int     `json:"version"`
	IsOpen    bool    `json:"is_open"`
}

// BalanceProjection maintains a map from account ID to current balance.
// It subscribes to the event stream and updates its state as events
// arrive. Queries against this projection are O(1) lookups.
//
// In a production system, this would be backed by a database table
// (e.g., a PostgreSQL table with an index on account_id) rather than
// an in-memory map. The subscription mechanism would be a durable
// consumer (polling the event store or consuming from a message
// broker) rather than an in-process callback.
type BalanceProjection struct {
	mu       sync.RWMutex
	accounts map[string]*AccountView
}

// NewBalanceProjection creates a new projection and subscribes it
// to the given event store. From this point forward, every event
// persisted to the store will update this projection.
func NewBalanceProjection(store *es.InMemoryEventStore) *BalanceProjection {
	p := &BalanceProjection{
		accounts: make(map[string]*AccountView),
	}

	store.Subscribe(p.handleEvent)
	return p
}

// GetAccount returns the current view of an account, or nil if
// the account does not exist in this projection.
func (p *BalanceProjection) GetAccount(accountID string) *AccountView {
	p.mu.RLock()
	defer p.mu.RUnlock()

	view, exists := p.accounts[accountID]
	if !exists {
		return nil
	}

	// Return a copy to prevent mutation of internal state.
	return new(*view)
}

// GetAllAccounts returns views of all known accounts, sorted by
// account ID for deterministic ordering.
func (p *BalanceProjection) GetAllAccounts() []AccountView {
	p.mu.RLock()
	defer p.mu.RUnlock()

	result := make([]AccountView, 0, len(p.accounts))
	for _, view := range p.accounts {
		result = append(result, *view)
	}

	// Sort by AccountID for deterministic output.
	sort.Slice(result, func(i, j int) bool {
		return result[i].AccountID < result[j].AccountID
	})

	return result
}

// handleEvent is the event handler that updates the projection.
// It examines each event and updates the appropriate account view.
//
// Note the pattern: the projection handles the same event types
// as the aggregate, but with a different purpose. The aggregate
// updates its internal state for business rule evaluation. The
// projection updates a query-optimized view for reading.
func (p *BalanceProjection) handleEvent(event es.Event) {
	if event.AggregateType != account.AggregateType {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	switch e := event.Data.(type) {
	case account.AccountOpenedEvent:
		p.accounts[event.AggregateID] = &AccountView{
			AccountID: event.AggregateID,
			Owner:     e.Owner,
			Balance:   e.InitialBalance,
			Version:   event.Version,
			IsOpen:    true,
		}

	case account.MoneyDepositedEvent:
		if view, ok := p.accounts[event.AggregateID]; ok {
			view.Balance += e.Amount
			view.Version = event.Version
		}

	case account.MoneyWithdrawnEvent:
		if view, ok := p.accounts[event.AggregateID]; ok {
			view.Balance -= e.Amount
			view.Version = event.Version
		}
	}
}
```

Now let us build a second projection that demonstrates a different query pattern: a transaction ledger.

Create `pkg/projection/ledger.go`:

```go
package projection

import (
	"sync"
	"time"

	"event-sourcing/pkg/account"
	"event-sourcing/pkg/es"
)

// Transaction represents a single entry in the transaction ledger.
type Transaction struct {
	EventID   string    `json:"event_id"`
	AccountID string    `json:"account_id"`
	Type      string    `json:"type"`
	Amount    float64   `json:"amount"`
	Reason    string    `json:"reason"`
	Balance   float64   `json:"balance_after"`
	Timestamp time.Time `json:"timestamp"`
}

// LedgerProjection maintains a chronological list of transactions
// for each account. It answers queries like "show me the last 10
// transactions for account X" efficiently.
//
// This projection demonstrates that the same events can power
// multiple read models. The BalanceProjection only stores the
// current balance; this one stores the full transaction history.
// Both consume the same event stream, but they serve different
// query patterns.
type LedgerProjection struct {
	mu           sync.RWMutex
	transactions map[string][]Transaction // accountID -> transactions
	balances     map[string]float64       // running balance per account
}

// NewLedgerProjection creates a ledger projection subscribed to
// the store.
func NewLedgerProjection(store *es.InMemoryEventStore) *LedgerProjection {
	p := &LedgerProjection{
		transactions: make(map[string][]Transaction),
		balances:     make(map[string]float64),
	}

	store.Subscribe(p.handleEvent)
	return p
}

// GetTransactions returns all transactions for an account in
// chronological order. Returns nil if the account is unknown.
func (p *LedgerProjection) GetTransactions(accountID string) []Transaction {
	p.mu.RLock()
	defer p.mu.RUnlock()

	transaction := p.transactions[accountID]
	if transaction == nil {
		return nil
	}

	// Return a copy.
	result := make([]Transaction, len(transaction))
	copy(result, transaction)
	return result
}

// GetRecentTransactions returns the last N transactions for an account.
func (p *LedgerProjection) GetRecentTransactions(accountID string, limit int) []Transaction {
	p.mu.RLock()
	defer p.mu.RUnlock()

	transaction := p.transactions[accountID]
	if transaction == nil {
		return nil
	}

	if limit >= len(transaction) {
		result := make([]Transaction, len(transaction))
		copy(result, transaction)
		return result
	}

	result := make([]Transaction, limit)
	copy(result, transaction[len(transaction)-limit:])
	return result
}

func (p *LedgerProjection) handleEvent(event es.Event) {
	if event.AggregateType != account.AggregateType {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	switch e := event.Data.(type) {
	case account.AccountOpenedEvent:
		p.balances[event.AggregateID] = e.InitialBalance
		if e.InitialBalance > 0 {
			p.transactions[event.AggregateID] = append(
				p.transactions[event.AggregateID],
				Transaction{
					EventID:   event.ID,
					AccountID: event.AggregateID,
					Type:      "deposit",
					Amount:    e.InitialBalance,
					Reason:    "initial_balance",
					Balance:   e.InitialBalance,
					Timestamp: event.Timestamp,
				},
			)
		}

	case account.MoneyDepositedEvent:
		p.balances[event.AggregateID] += e.Amount
		p.transactions[event.AggregateID] = append(
			p.transactions[event.AggregateID],
			Transaction{
				EventID:   event.ID,
				AccountID: event.AggregateID,
				Type:      "deposit",
				Amount:    e.Amount,
				Reason:    e.Reason,
				Balance:   p.balances[event.AggregateID],
				Timestamp: event.Timestamp,
			},
		)

	case account.MoneyWithdrawnEvent:
		p.balances[event.AggregateID] -= e.Amount
		p.transactions[event.AggregateID] = append(
			p.transactions[event.AggregateID],
			Transaction{
				EventID:   event.ID,
				AccountID: event.AggregateID,
				Type:      "withdrawal",
				Amount:    e.Amount,
				Reason:    e.Reason,
				Balance:   p.balances[event.AggregateID],
				Timestamp: event.Timestamp,
			},
		)
	}
}
```

Two projections, same event stream, completely different data structures.
This is the power of CQRS: you can add a new query pattern (say, `"accounts with high activity"` for fraud detection) by creating a new projection that subscribes to the same events, without touching the write model at all.

The ledger projection also demonstrates a technique that the balance projection does not: it maintains a running balance in its internal state (`p.balances`) and includes the `balance_after` field in each transaction record.
This denormalized data makes it possible to show the balance at any point in the transaction history without re-computing it from scratch.

---

#### Step 7: Snapshotting - taming long event streams

An account that has been active for years might accumulate thousands of events. Replaying all of them every time the account is loaded becomes expensive.
Snapshotting solves this by periodically saving a checkpoint of the aggregate's state.
When loading the aggregate, you load the most recent snapshot and then replay only the events that occurred after the snapshot was taken.

Create `pkg/es/snapshot.go`:

```go
package es

import (
	"sync"
	"time"
)

// Snapshot represents a point in time capture of an aggregate's state.
// Instead of replaying all events from the beginning, the system can
// load the most recent snapshot and replay only events since that
// version.
//
// For an account with 10,000 events and a snapshot at version 9,950,
// loading the aggregate requires deserializing one snapshot and
// replaying 50 events, rather than replaying all 10,000.
type Snapshot struct {
	AggregateID   string
	AggregateType string
	Version       int // the event version this snapshot represents
	State         any // the serialized aggregate state
	Timestamp     time.Time
}

// SnapshotStore defines the interface for persisting and retrieving
// snapshots. In production, this would be backed by a database
// (often the same one as the event store, in a separate table).
type SnapshotStore interface {
	SaveSnapshot(snapshot Snapshot) error
	LoadSnapshot(aggregateID string) (*Snapshot, error)
}

// InMemorySnapshotStore is a thread-safe, in-memory snapshot store.
type InMemorySnapshotStore struct {
	mu        sync.RWMutex
	snapshots map[string]Snapshot
}

// NewInMemorySnapshotStore creates a new empty snapshot store.
func NewInMemorySnapshotStore() *InMemorySnapshotStore {
	return &InMemorySnapshotStore{
		snapshots: make(map[string]Snapshot),
	}
}

// SaveSnapshot stores (or replaces) the snapshot for an aggregate.
func (s *InMemorySnapshotStore) SaveSnapshot(snapshot Snapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	snapshot.Timestamp = time.Now()
	s.snapshots[snapshot.AggregateID] = snapshot
	return nil
}

// LoadSnapshot retrieves the most recent snapshot for an aggregate.
// Returns nil if no snapshot exists.
func (s *InMemorySnapshotStore) LoadSnapshot(aggregateID string) (*Snapshot, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	snap, exists := s.snapshots[aggregateID]
	if !exists {
		return nil, nil
	}
	return &snap, nil
}
```

Now let us add snapshot support to the account aggregate.

Create `pkg/account/snapshot.go`:

```go
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
```

The snapshot strategy is transparent to the aggregate. The `Account` does not know whether its state came from a full replay or from a snapshot plus partial replay.
It exposes `TakeSnapshot` and `LoadFromSnapshot`, and the command handler decides when to call them.

A typical snapshot interval is 50 to 100 events. This is a tradeoff: more frequent snapshots mean faster loading but more storage and write overhead; less frequent snapshots save storage but increase load time.
The optimal interval depends on your event rate, aggregate load frequency, and performance requirements.

---

#### Step 8: Testing event-sourced systems

We now have a complete write side (events, event store, aggregate, command handler, snapshots) and a complete read side (balance and ledger projections).
Before we wire everything together in a final demo, let us build a comprehensive test suite.

Event-sourced systems have a distinctive testing advantage: because all state changes are captured as events,
you can write tests in a `"given-when-then"` style that is both expressive and precise.

- `"Given"` = these events have already occurred (the aggregate's history).
- `"When"` = this command is executed.
- `"Then"` = these new events should be produced (or this error should be returned).

This approach tests business logic without needing a database, an event store, or any infrastructure at all.
Let us build the tests layer by layer, starting from the innermost component (the aggregate) and working outward.

**Aggregate tests - pure business logic, no infrastructure**

Create `pkg/account/aggregate_test.go`. These tests exercise the aggregate in complete isolation.
There is no event store, no command handler - just an aggregate struct, some events, and assertions.
This is possible because the aggregate's state is derived entirely from events, which means you can construct any scenario by replaying a specific sequence of events.

A small helper makes this pattern clean:

```go
// openAccount returns an Account that has been opened with the given
// owner and initial balance, ready for further operations.
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
```

With this helper, every test reads like a business requirement. Here is the overdraft protection test:

```go
func TestWithdraw_InsufficientBalance(t *testing.T) {
	// Given: an open account with $50.
	a := openAccount("acc-001", "Alice", 50.0)

	// When: we attempt to withdraw $100.
	err := a.Withdraw(100.0, "rent")

	// Then: error, no new events, balance unchanged.
	if err == nil {
		t.Fatal("expected error for insufficient balance")
	}
	if len(a.Changes()) != 0 {
		t.Errorf("expected 0 changes after rejection, got %d", len(a.Changes()))
	}
	if a.Balance() != 50.0 {
		t.Errorf("balance should remain 50.00, got %.2f", a.Balance())
	}
}
```

Notice what this test verifies beyond just the error: it confirms that no events were produced and that the balance was not modified.
In event sourcing, a rejected command must leave zero trace - no events, no state changes.
Testing both conditions catches subtle defects where the aggregate might raise an event before discovering the validation failure.

Our aggregate test file should cover the full range of scenarios:
- opening accounts (success, already open, empty owner, negative balance)
- deposits (success, zero amount, negative amount, account not open)
- withdrawals (success, exact balance, insufficient funds, account not open)
- the dual-path apply mechanism (historical replay produces no changes, new commands produce changes)
- version tracking, and the cumulative state after a mixed sequence of operations

One test that deserves particular attention verifies the fundamental contract of event sourcing - that the aggregate's state is identical whether
it was built by issuing commands one by one or by replaying those same commands as historical events:

```go
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
	if replayed.Balance() != live.Balance() {
		t.Errorf("balance mismatch: live=%.2f, replayed=%.2f",
			live.Balance(), replayed.Balance())
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
}
```

If this test passes, you know that events are a complete and faithful record of state transitions, and that replaying them always reproduces the same state.

**Event store tests - infrastructure correctness**

Create `pkg/es/store_test.go`. These tests verify the event store independently of any domain logic.
They should confirm that the store assigns sequential versions, returns defensive copies (so callers cannot corrupt internal state),
rejects stale writes via concurrency checks, isolates events by aggregate ID, correctly handles `LoadEventsFrom` for partial replay, and delivers events to subscribers in order.

These infrastructure tests should use generic payloads (strings, maps) rather than domain events.
This is deliberate: the event store should not know or care about the domain. It stores and retrieves events; the domain gives them meaning.

Similarly, create `pkg/es/snapshot_test.go` to verify the snapshot store.
Key tests include the basic save and load round-trip, overwrite semantics (saving a new snapshot replaces the old one),
aggregate isolation (Alice's snapshot does not affect Bob's), automatic timestamp assignment, and thread safety under concurrent access.

**Command handler tests - the full pipeline**

Create `pkg/account/commands_test.go`. These tests exercise the command handler pipeline end to end: load events from the store,
rebuild the aggregate, execute the command, and save the resulting events.
They use a real (in-memory) event store, so they verify the integration between the handler and the store, including optimistic concurrency control.

A few helpers keep the tests concise:

```go
func setupHandler() (*es.InMemoryEventStore, *CommandHandler) {
	store := es.NewInMemoryEventStore()
	handler := NewCommandHandler(store)
	return store, handler
}

func openTestAccount(t *testing.T, handler *CommandHandler, id, owner string, balance float64) {
	t.Helper()
	err := handler.HandleOpenAccount(OpenAccountCommand{
		AccountID: id, Owner: owner, InitialBalance: balance,
	})
	if err != nil {
		t.Fatalf("failed to open account %s: %v", id, err)
	}
}

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
```

The `accountBalance` helper is worth noting: it verifies the persisted state by loading events from the store and replaying them, rather than trusting any in-memory cache.
This is the strongest form of assertion in an event-sourced system - if the events in the store produce the correct balance when replayed, the system is correct.

The concurrency test is particularly important because it demonstrates the core safety mechanism that prevents the `"phantom money"` defect described in the introduction:

```go
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

	// Only one withdrawal should have been persisted. The balance
	// should be 1000 - 800 = 200, not 1000 - 800 - 800 = -600.
	if accountBalance(t, store, "acc-001") != 200.0 {
		t.Errorf("expected balance 200.00 (one withdrawal), got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}
```

Two goroutines both load the same account at version 1, both attempt a `$800` withdrawal against a `$1000` balance, and both pass the `"sufficient funds"` check locally.
But only the first save succeeds. The second is rejected by the event store's version check, preventing the account from going to `-$600`.

The command handler tests should also cover the transfer workflow across two aggregates.
One particularly instructive test documents the known limitation where a failed deposit leaves the source account debited:

```go
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
	if accountBalance(t, store, "acc-001") != 400.0 {
		t.Errorf("alice: expected 400.00 (debited), got %.2f",
			accountBalance(t, store, "acc-001"))
	}
}
```

This test documents the system's behavior rather than asserting an ideal outcome.
In a production system, a Saga would emit a compensating event to reverse the withdrawal.
Here, the test makes the limitation explicit and verifiable.

Our command handler test file should also include a complex workflow test that exercises every handler method in combination (open accounts, deposit, withdraw, transfer)
and verifies conservation of money - the total balance across all accounts should equal the sum of external deposits minus external withdrawals:

```go
func TestCrossOperation_ComplexWorkflow(t *testing.T) {
	store, handler := setupHandler()

	openTestAccount(t, handler, "acc-A", "Alice", 1000)
	openTestAccount(t, handler, "acc-B", "Bob", 500)
	openTestAccount(t, handler, "acc-C", "Carol", 200)

	_ = handler.HandleDeposit(DepositCommand{
		AccountID: "acc-A", Amount: 200, Reason: "bonus",
	})
	_ = handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-A", ToAccountID: "acc-B", Amount: 300,
	})
	_ = handler.HandleWithdraw(WithdrawCommand{
		AccountID: "acc-B", Amount: 100, Reason: "dinner",
	})
	_ = handler.HandleTransfer(TransferCommand{
		FromAccountID: "acc-B", ToAccountID: "acc-C", Amount: 150,
	})
	_ = handler.HandleDeposit(DepositCommand{
		AccountID: "acc-C", Amount: 50, Reason: "refund",
	})

	// Conservation of money: total should be 1000+500+200 (initial)
	// + 200+50 (external deposits) - 100 (external withdrawal) = 1850.
	totalBalance := accountBalance(t, store, "acc-A") +
		accountBalance(t, store, "acc-B") +
		accountBalance(t, store, "acc-C")
	if totalBalance != 1850.0 {
		t.Errorf("conservation of money violated: expected 1850.00, got %.2f",
			totalBalance)
	}
}
```

This conservation check is a powerful invariant test. Transfers move money between accounts but should never create or destroy it.
If this test fails, there is a fundamental defect in the system.

**Snapshot tests - verifying the optimization path**

Create `pkg/account/snapshot_test.go`. These tests verify the snapshot mechanism and the snapshot aware command handler.
The most critical property to test is that an aggregate restored from a snapshot plus partial event replay produces the exact same state as a full event replay from scratch:

```go
func TestSnapshotRoundtrip_FullReplayEquivalence(t *testing.T) {
	allEvents := []es.Event{
		// ... five events: open, deposit, withdraw, deposit, withdraw
	}

	// Path A: full replay.
	fullReplay := NewAccount("acc-001")
	fullReplay.LoadFromHistory(allEvents)

	// Path B: snapshot at version 3, then replay events 4 and 5.
	snapshotAt3 := NewAccount("acc-001")
	snapshotAt3.LoadFromHistory(allEvents[:3])
	snap := snapshotAt3.TakeSnapshot()

	fromSnap := NewAccount("acc-001")
	_ = fromSnap.LoadFromSnapshot(snap)
	fromSnap.LoadFromHistory(allEvents[3:])

	// Both paths should produce identical state.
	if fullReplay.Balance() != fromSnap.Balance() {
		t.Errorf("Balance mismatch: full=%.2f, snap=%.2f",
			fullReplay.Balance(), fromSnap.Balance())
	}
	if fullReplay.Version() != fromSnap.Version() {
		t.Errorf("Version mismatch: full=%d, snap=%d",
			fullReplay.Version(), fromSnap.Version())
	}
}
```

If this test passes, you know that the snapshot is a faithful representation of the aggregate's state and that the partial replay correctly resumes from the snapshot point.

Additional snapshot tests should verify that the `SnapshotCommandHandler` takes snapshots at the configured interval,
that it falls back to full replay when no snapshot exists, that it loads from the snapshot and replays only newer events when a snapshot is available,
and that business rules (like overdraft protection) still work correctly regardless of whether the aggregate was loaded via snapshot or full replay.

**Projection tests - verifying the read side**

Create `pkg/projection/balance_test.go` and `pkg/projection/ledger_test.go`.
These tests verify that projections correctly transform events into query optimized views.
They should use the event store's `SaveEvents` method (rather than calling the projection's event handler directly) to ensure the subscriber wiring is exercised.

A helper that persists events through the store keeps the tests focused:

```go
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
```

Balance projection tests should cover account creation (with and without initial balance), deposit and withdrawal updates,
querying unknown accounts, returning defensive copies, tracking multiple accounts independently, the `GetAllAccounts` sort order,
and ignoring events from other aggregate types.

Ledger projection tests should verify the running balance column across a sequence of operations, the `GetRecentTransactions` query,
correct transaction entries for transfers (with `"transfer_out"` and `"transfer_in"` reasons), and that zero balance account openings produce no ledger entry.

**Running the full suite**

To run all the tests across every package:

```bash
go test -v ./...
```

You should see tests passing in four packages: `pkg/es` (event store and snapshot store), `pkg/account` (aggregate, command handler, and snapshot handler),
and `pkg/projection` (balance and ledger projections).
Together, the test suite covers the core properties that matter in an event-sourced system: events are the source of truth, state is derived correctly from events,
concurrent modifications are detected, projections stay in sync with the event stream, and snapshots are a transparent optimization.

---

#### Step 9: Putting it all together

Let us build a complete example that demonstrates the entire system working end to end.

Create `main.go` in the project root:

```go
package main

import (
	"event-sourcing/pkg/account"
	"event-sourcing/pkg/es"
	"event-sourcing/pkg/projection"
	"fmt"
)

func main() {
	// Initialize the infrastructure.
	eventStore := es.NewInMemoryEventStore()
	snapStore := es.NewInMemorySnapshotStore()

	// Create projections. They subscribe to the event store
	// automatically upon creation.
	balanceView := projection.NewBalanceProjection(eventStore)
	ledgerView := projection.NewLedgerProjection(eventStore)

	// Create command handlers: one with snapshot support (every 10
	// events), and a base handler for operations that the snapshot
	// handler does not cover (like opening accounts).
	cmdHandler := account.NewSnapshotCommandHandler(eventStore, snapStore, 10)
	baseHandler := account.NewCommandHandler(eventStore)

	fmt.Println("=== Event Sourcing & CQRS Demo ===")
	fmt.Println()

	// Open accounts
	fmt.Println("--- Opening accounts ---")
	must(baseHandler.HandleOpenAccount(account.OpenAccountCommand{
		AccountID: "acc-alice", Owner: "Alice", InitialBalance: 1000,
	}))
	must(baseHandler.HandleOpenAccount(account.OpenAccountCommand{
		AccountID: "acc-bob", Owner: "Bob", InitialBalance: 500,
	}))
	printBalances(balanceView)

	// Perform some operations
	fmt.Println("--- Alice deposits $500 ---")
	must(cmdHandler.HandleDeposit(account.DepositCommand{
		AccountID: "acc-alice", Amount: 500, Reason: "paycheck",
	}))
	printBalances(balanceView)

	fmt.Println("--- Bob withdraws $200 ---")
	must(cmdHandler.HandleWithdraw(account.WithdrawCommand{
		AccountID: "acc-bob", Amount: 200, Reason: "rent",
	}))
	printBalances(balanceView)

	// Transfer between accounts
	fmt.Println("--- Transfer $300 from Alice to Bob ---")
	must(baseHandler.HandleTransfer(account.TransferCommand{
		FromAccountID: "acc-alice", ToAccountID: "acc-bob", Amount: 300,
	}))
	printBalances(balanceView)

	// Show the event history (the source of truth)
	fmt.Println("--- Alice's event history ---")
	events, _ := eventStore.LoadEvents("acc-alice")
	for _, evt := range events {
		fmt.Printf("  [v%d] %s: %+v\n",
			evt.Version, evt.EventType, evt.Data)
	}
	fmt.Println()

	// Show the transaction ledger (a projection)
	fmt.Println("--- Alice's transaction ledger ---")
	transactions := ledgerView.GetTransactions("acc-alice")
	for _, txn := range transactions {
		sign := "+"
		if txn.Type == "withdrawal" {
			sign = "-"
		}
		fmt.Printf("  %s$%.2f  %-15s  balance: $%.2f\n",
			sign, txn.Amount, txn.Reason, txn.Balance)
	}
	fmt.Println()

	// Demonstrate temporal query
	// Rebuild Alice's state at version 2 (after opening + first deposit).
	fmt.Println("--- Alice's balance after opening (version 1) ---")
	historicalEvents, _ := eventStore.LoadEvents("acc-alice")
	historicalAccount := account.NewAccount("acc-alice")
	for _, evt := range historicalEvents {
		if evt.Version > 1 {
			break
		}
		historicalAccount.LoadFromHistory([]es.Event{evt})
	}
	fmt.Printf("  Balance at v1: $%.2f (owner: %s)\n",
		historicalAccount.Balance(), historicalAccount.Owner())
	fmt.Println()

	// Demonstrate error handling
	fmt.Println("--- Attempting invalid withdrawal ---")
	err := cmdHandler.HandleWithdraw(account.WithdrawCommand{
		AccountID: "acc-bob", Amount: 9999, Reason: "too-much",
	})
	if err != nil {
		fmt.Printf("  Correctly rejected: %v\n", err)
	}
	fmt.Println()

	// Final state
	fmt.Println("--- Final state ---")
	printBalances(balanceView)
}

func printBalances(view *projection.BalanceProjection) {
	for _, acct := range view.GetAllAccounts() {
		fmt.Printf("  %s (%s): $%.2f\n",
			acct.AccountID, acct.Owner, acct.Balance)
	}
	fmt.Println()
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
```

Run the full example:

```bash
go run main.go
```

And run all the tests:

```bash
go test -v ./...
```

---

#### Conclusion

Throughout this deep dive, we built a complete bank account system using event sourcing and CQRS in Go, starting from the foundational concepts and working up to snapshots, projections, and comprehensive tests.

The key takeaways from this experience are worth revisiting.

- **Events are facts, not intentions.** The sharp distinction between commands (what someone wants to happen) and events (what actually happened) is more than a naming convention.
  It changes how you think about system design. Commands can be rejected; events cannot. Commands express intent; events express truth.
  This distinction makes the system's behavior transparent: you can read the event stream and understand exactly what happened, when, and in what order.

- **State is derived, never stored directly.** The aggregate's balance field is a cache, not the source of truth. If you delete the field and replay the events, you get the same number.
  This property has profound implications: you can rebuild any projection from scratch by replaying the event stream, you can fix defects in projection logic and backfill correct data,
  and you can add entirely new query patterns retroactively by creating a new projection that replays all historical events.

- **Optimistic concurrency is the safety net.** Without concurrency control, event sourcing is vulnerable to the same lost update problems as any system.
  The `expectedVersion` check in `SaveEvents` is the mechanism that prevents two concurrent command handlers from both succeeding when only one should.
  In our test, this prevented a scenario where two `$400` withdrawals would both succeed against a `$500` balance.

- **Projections decouple reading from writing.** The balance projection and the ledger projection consume the same events but serve completely different query patterns.
  Adding a new projection (say, `"accounts with high activity"` for fraud detection) requires no changes to the command side.
  This is the practical benefit of CQRS: your read models can evolve independently of your write model.

- **Snapshotting is an optimization, not a requirement.** The system works correctly without snapshots - they just make aggregate loading faster for long event streams.
  The snapshot aware command handler transparently falls back to full replay when no snapshot exists, and the aggregate does not know or care whether its state came from a snapshot or from events.

- **Testing event-sourced systems is remarkably clean.** The `"given events, when command, then events"` pattern maps directly to business requirements and produces tests that are readable by non-engineers.
  The aggregate tests need no database, no event store, no infrastructure at all - just events in, events out.

For production use, consider extending this foundation with:

- **Persistent storage** for both the event store and snapshot store - PostgreSQL with a well indexed events table is a common choice.
- **Event serialization** using JSON or Protobuf for durable storage.
- **Event versioning and upcasting** strategies for evolving event schemas without breaking old events.
- **Process managers or sagas** for complex workflows that span multiple aggregates, such as our transfer operation.
- **Integration with message brokers** like NATS, RabbitMQ, or Kafka for asynchronous projection updates and cross service event distribution.

**Further reading:**

- [Microsoft - Event Sourcing pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing)
- [Microsoft - CQRS pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs)
