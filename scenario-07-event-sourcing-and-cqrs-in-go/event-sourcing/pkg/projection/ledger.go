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
