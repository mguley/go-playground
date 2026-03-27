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
