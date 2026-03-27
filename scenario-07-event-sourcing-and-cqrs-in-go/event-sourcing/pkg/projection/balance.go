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
