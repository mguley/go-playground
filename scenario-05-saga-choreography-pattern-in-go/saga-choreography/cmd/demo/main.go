package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"

	"saga-choreography/internal/inventory"
	"saga-choreography/internal/monitor"
	"saga-choreography/internal/order"
	"saga-choreography/internal/payment"
	"saga-choreography/internal/shipping"
	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

func main() {
	// Create logger
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	// Create event bus
	eventBus := events.NewEventBus(logger)

	// Create the saga monitor before services so it captures all events.
	// The monitor is a passive observer and subscribes to every event type,
	// so it must be created first to ensure nothing is missed.
	sagaMonitor := monitor.NewSagaMonitor(eventBus, logger)

	// Create repositories
	orderRepo := order.NewRepository()
	paymentRepo := payment.NewRepository()
	inventoryRepo := inventory.NewRepository()
	shippingRepo := shipping.NewRepository()

	// Create services. Each service subscribes to its relevant events during construction.
	// The order in which services are created doesn't matter for correctness,
	// but the monitor should be created first to ensure it captures everything.
	orderService := order.NewService(orderRepo, eventBus, logger)
	_ = payment.NewService(paymentRepo, eventBus, logger)
	_ = inventory.NewService(inventoryRepo, eventBus, logger)
	shippingService := shipping.NewService(shippingRepo, eventBus, logger)

	ctx := context.Background()
	sagaTimeout := 5 * time.Second

	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("SAGA PATTERN DEMO: CHOREOGRAPHY-BASED DISTRIBUTED TRANSACTIONS")
	fmt.Println("================================================================================")

	fmt.Println()
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println("SCENARIO 1: Successful order flow")
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println()

	items := []models.OrderItem{
		{ProductID: "PROD-001", Quantity: 1, Price: 999.99},
		{ProductID: "PROD-002", Quantity: 2, Price: 29.99},
	}

	newOrder, err := orderService.CreateOrder(ctx, "CUST-003", items)
	if err != nil {
		logger.Error("Failed to create order: %v", err)
		os.Exit(1)
	}

	// Wait for the saga to complete instead of sleeping an arbitrary duration
	if err = eventBus.WaitForSaga(ctx, sagaTimeout); err != nil {
		logger.Error("Saga timeout: %v", err)
		os.Exit(1)
	}

	finalOrder, _ := orderService.GetOrder(newOrder.ID)
	fmt.Printf("\n>>> Final Order Status: %s\n", finalOrder.Status)
	sagaMonitor.PrintSagaTimeline(newOrder.ID)
	eventBus.Reset() // Clear completion signal for next saga

	fmt.Println()
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println("SCENARIO 2: Payment failure (insufficient funds)")
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println()

	expensiveItems := []models.OrderItem{
		{ProductID: "PROD-001", Quantity: 10, Price: 999.99}, // $9999.90 - exceeds CUST-002 balance
	}

	newOrder2, err := orderService.CreateOrder(ctx, "CUST-002", expensiveItems)
	if err != nil {
		logger.Error("Failed to create order: %v", err)
		os.Exit(1)
	}

	if err = eventBus.WaitForSaga(ctx, sagaTimeout); err != nil {
		logger.Error("Saga timeout: %v", err)
		os.Exit(1)
	}

	finalOrder2, _ := orderService.GetOrder(newOrder2.ID)
	fmt.Printf("\n>>> Final Order Status: %s\n", finalOrder2.Status)
	sagaMonitor.PrintSagaTimeline(newOrder2.ID)
	eventBus.Reset()

	fmt.Println()
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println("SCENARIO 3: Shipping failure (triggers full compensation)")
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println()

	// Enable shipping failure simulation
	shippingService.SetSimulateFailure(true)

	items3 := []models.OrderItem{
		{ProductID: "PROD-002", Quantity: 1, Price: 29.99},
	}

	newOrder3, err := orderService.CreateOrder(ctx, "CUST-003", items3)
	if err != nil {
		logger.Error("Failed to create order: %v", err)
		os.Exit(1)
	}

	if err = eventBus.WaitForSaga(ctx, sagaTimeout); err != nil {
		logger.Error("Saga timeout: %v", err)
		os.Exit(1)
	}

	finalOrder3, _ := orderService.GetOrder(newOrder3.ID)
	fmt.Printf("\n>>> Final Order Status: %s\n", finalOrder3.Status)
	sagaMonitor.PrintSagaTimeline(newOrder3.ID)

	// Disable failure simulation for any future use
	shippingService.SetSimulateFailure(false)
	eventBus.Reset()

	fmt.Println()
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println("SCENARIO 4: Inventory failure (out-of-stock item)")
	fmt.Println("────────────────────────────────────────────────────────────────────────────────")
	fmt.Println()

	oosItems := []models.OrderItem{
		{ProductID: "PROD-004", Quantity: 1, Price: 149.99}, // PROD-004 has 0 stock
	}

	newOrder4, err := orderService.CreateOrder(ctx, "CUST-001", oosItems)
	if err != nil {
		logger.Error("Failed to create order: %v", err)
		os.Exit(1)
	}

	if err = eventBus.WaitForSaga(ctx, sagaTimeout); err != nil {
		logger.Error("Saga timeout: %v", err)
		os.Exit(1)
	}

	finalOrder4, _ := orderService.GetOrder(newOrder4.ID)
	fmt.Printf("\n>>> Final Order Status: %s\n", finalOrder4.Status)
	sagaMonitor.PrintSagaTimeline(newOrder4.ID)

	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("MONITORING DASHBOARD")
	fmt.Println("================================================================================")

	metrics := sagaMonitor.GetMetrics()
	fmt.Println()
	fmt.Printf("  Total Sagas:      %d\n", metrics.TotalSagas)
	fmt.Printf("  Completed:        %d\n", metrics.CompletedSagas)
	fmt.Printf("  Cancelled:        %d\n", metrics.CancelledSagas)
	fmt.Printf("  In Progress:      %d\n", metrics.InProgressSagas)
	fmt.Printf("  Avg Duration:     %v\n", metrics.AverageDuration)

	// Check for stuck sagas (there should be none in our demo)
	stuckSagas := sagaMonitor.GetStuckSagas(3 * time.Second)
	if len(stuckSagas) > 0 {
		fmt.Printf("\n  WARNING: %d stuck saga(s) detected!\n", len(stuckSagas))
	}
}
