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
