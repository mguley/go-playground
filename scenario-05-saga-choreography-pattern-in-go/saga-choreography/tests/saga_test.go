package tests

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"testing"
	"time"

	"saga-choreography/internal/inventory"
	"saga-choreography/internal/monitor"
	"saga-choreography/internal/order"
	"saga-choreography/internal/payment"
	"saga-choreography/internal/shipping"
	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// testHarness bundles all the components needed for saga integration tests.
// Each test creates a fresh harness, ensuring complete isolation between tests.
// We keep references to every repository so we can inspect the internal state
// of each service after the saga runs - this is what lets us verify that
// compensating transactions actually did their job.
type testHarness struct {
	ctx      context.Context
	eventBus *events.EventBus
	monitor  *monitor.SagaMonitor

	orderRepo     *order.Repository
	paymentRepo   *payment.Repository
	inventoryRepo *inventory.Repository
	shippingRepo  *shipping.Repository

	orderSvc    *order.Service
	shippingSvc *shipping.Service
}

// newSyncTestHarness creates a completely fresh set of services with synchronous
// event delivery for deterministic test execution.
//
// With synchronous delivery, when an event is published, each subscriber's handler
// runs to completion - including any events that handler publishes - before the
// next subscriber is notified. This makes the entire saga complete within the
// CreateOrder call, eliminating the need for sleeps or polling.
//
// The tradeoff is that subscriber registration order matters: Inventory and Shipping
// must subscribe to "order.created" before Payment, because synchronous fan-out
// processes subscribers sequentially. If Payment runs first, it publishes
// "payment.completed" before Inventory has cached the order items, causing a
// spurious failure.
func newSyncTestHarness(t *testing.T) *testHarness {
	t.Helper()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	bus := events.NewEventBus(logger)

	// Synchronous delivery ensures the entire saga (including all compensation)
	// completes within the CreateOrder call. No sleeps or polling needed.
	bus.SetAsyncDelivery(false)

	// Create the monitor first so it captures every event from the start.
	mon := monitor.NewSagaMonitor(bus, logger)

	// Create repositories - each comes pre-seeded with test data.
	orderRepo := order.NewRepository()
	paymentRepo := payment.NewRepository()
	inventoryRepo := inventory.NewRepository()
	shippingRepo := shipping.NewRepository()

	// Create the Order service first (subscribes to downstream events).
	oSvc := order.NewService(orderRepo, bus, logger)

	// IMPORTANT: Create Inventory and Shipping services BEFORE Payment.
	//
	// With synchronous delivery, subscribers are notified in registration order.
	// Inventory and Shipping need to receive "order.created" first (to cache
	// order items and customer data) before Payment processes the event and
	// publishes "payment.completed", which triggers the next step.
	//
	// In production with async delivery, this ordering doesn't matter because
	// all subscribers process "order.created" concurrently, and there's enough
	// time for caches to be populated before downstream events arrive.
	_ = inventory.NewService(inventoryRepo, bus, logger)
	sSvc := shipping.NewService(shippingRepo, bus, logger)
	_ = payment.NewService(paymentRepo, bus, logger)

	return &testHarness{
		ctx:           context.Background(),
		eventBus:      bus,
		monitor:       mon,
		orderRepo:     orderRepo,
		paymentRepo:   paymentRepo,
		inventoryRepo: inventoryRepo,
		shippingRepo:  shippingRepo,
		orderSvc:      oSvc,
		shippingSvc:   sSvc,
	}
}

// newAsyncTestHarness creates a fresh set of services with asynchronous event
// delivery, mirroring real production behavior.
//
// In async mode, every subscriber receives events in its own goroutine.
// This means:
//   - Events are processed concurrently across services.
//   - Subscriber registration order is irrelevant (all goroutines run in parallel).
//   - The saga does NOT complete within the CreateOrder call. Tests must wait
//     for the saga to reach a terminal state, and then poll for compensation
//     events to settle.
//   - Race conditions between concurrent handlers are possible and must be
//     handled correctly by the service implementations (mutexes, atomic
//     operations, idempotency guards).
//
// This harness deliberately creates services in a different order than the
// sync harness to prove that registration order is irrelevant in async mode.
func newAsyncTestHarness(t *testing.T) *testHarness {
	t.Helper()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	bus := events.NewEventBus(logger)

	// Async delivery is the default (bus.asyncDelivery starts as true),
	// but we set it explicitly here to make the intent crystal clear.
	bus.SetAsyncDelivery(true)

	mon := monitor.NewSagaMonitor(bus, logger)

	orderRepo := order.NewRepository()
	paymentRepo := payment.NewRepository()
	inventoryRepo := inventory.NewRepository()
	shippingRepo := shipping.NewRepository()

	// Deliberately create services in a DIFFERENT order than the sync harness.
	// In async mode, all subscribers receive each event concurrently in separate
	// goroutines, so registration order cannot affect correctness. Creating
	// services in a different order here proves this property.
	oSvc := order.NewService(orderRepo, bus, logger)
	_ = payment.NewService(paymentRepo, bus, logger)       // Payment first this time
	_ = inventory.NewService(inventoryRepo, bus, logger)   // then Inventory
	sSvc := shipping.NewService(shippingRepo, bus, logger) // then Shipping

	return &testHarness{
		ctx:           context.Background(),
		eventBus:      bus,
		monitor:       mon,
		orderRepo:     orderRepo,
		paymentRepo:   paymentRepo,
		inventoryRepo: inventoryRepo,
		shippingRepo:  shippingRepo,
		orderSvc:      oSvc,
		shippingSvc:   sSvc,
	}
}

// almostEqual compares two float64 values within a small tolerance.
// Necessary because floating-point arithmetic (e.g., 5000.00 - 29.99 + 29.99)
// may not produce exactly the original value.
func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 0.001
}

// waitForCondition polls a condition function until it returns true or the
// timeout expires. This is the standard pattern for asserting eventual
// consistency in async integration tests.
//
// In a choreography-based saga with async event delivery, there's an inherent
// gap between the terminal event (order.completed or order.cancelled) and the
// completion of all downstream effects (refunds, inventory releases). The
// WaitForSaga method on the event bus blocks until the terminal event, but
// compensation events triggered by order.cancelled are still in flight when
// WaitForSaga returns. This helper bridges that gap by repeatedly checking
// the expected final state until it materializes or the timeout fires.
//
// The poll interval of 10ms is a balance between responsiveness (catching the
// state change quickly) and CPU usage (not busy-looping). In practice, most
// conditions are met within 50-100ms in the in-memory event bus.
func waitForCondition(t *testing.T, timeout time.Duration, condition func() bool, description string) {
	t.Helper()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out after %v waiting for: %s", timeout, description)
}

// sagaTimeout is the maximum time we're willing to wait for a saga to reach
// a terminal state (order.completed or order.cancelled). In the in-memory
// event bus, this is essentially instantaneous, but we set a generous timeout
// to avoid flaky tests on slow CI machines.
const sagaTimeout = 5 * time.Second

// compensationTimeout is the maximum time we'll poll for compensation events
// to settle after the saga reaches a terminal state. Compensation happens after
// order.cancelled is published, so we need extra time beyond the terminal event.
const compensationTimeout = 3 * time.Second

// ============================================================================
// Part A: Synchronous delivery tests
// ============================================================================
//
// These tests use synchronous event delivery where each subscriber's handler
// runs to completion (including publishing follow-on events) before the next
// subscriber is notified. This makes the entire saga deterministic and lets
// us assert immediately after CreateOrder returns.
//
// Sync tests validate business logic correctness: do the right events fire,
// do compensating transactions execute, do state changes land correctly?

// ---------------------------------------------------------------------------
// Test 1: Successful Order Saga (Happy Path)
// ---------------------------------------------------------------------------
// This test verifies that when everything goes right, the saga:
// - Completes the order with all saga references populated
// - Deducts the correct amount from the customer's balance
// - Reduces product stock by the ordered quantity
// - Creates a confirmed inventory reservation
// - Creates a scheduled shipment with a tracking number
// - Records the saga as COMPLETED in the monitor

func TestSuccessfulOrderSaga(t *testing.T) {
	h := newSyncTestHarness(t)

	// Capture initial state so we can verify deltas after the saga runs.
	// CUST-003 has $5000 balance; PROD-002 has 100 units in stock.
	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-003")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-002")

	// Place an order well within the customer's balance.
	items := []models.OrderItem{
		{ProductID: "PROD-002", Quantity: 3, Price: 29.99},
	}
	totalAmount := 3 * 29.99

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-003", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	// With synchronous delivery, the entire saga - including all event handlers
	// and any follow-on events they publish - completes within CreateOrder.
	// No waiting is needed; we can assert immediately.
	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCompleted {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCompleted)
	}
	if finalOrder.PaymentID == "" {
		t.Error("order PaymentID is empty; expected a payment reference")
	}
	if finalOrder.ReservationID == "" {
		t.Error("order ReservationID is empty; expected a reservation reference")
	}
	if finalOrder.ShipmentID == "" {
		t.Error("order ShipmentID is empty; expected a shipment reference")
	}

	// --- Payment assertions ---
	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusCompleted {
		t.Errorf("payment status = %s, want %s", pay.Status, models.PaymentStatusCompleted)
	}

	// Customer balance should be reduced by the exact order total.
	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-003")
	expectedBalance := customerBefore.Balance - totalAmount
	if !almostEqual(customerAfter.Balance, expectedBalance) {
		t.Errorf("customer balance = %.2f, want %.2f", customerAfter.Balance, expectedBalance)
	}

	// --- Inventory assertions ---
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-002")
	expectedQty := productBefore.Quantity - 3
	if productAfter.Quantity != expectedQty {
		t.Errorf("product PROD-002 quantity = %d, want %d", productAfter.Quantity, expectedQty)
	}

	reservation, err := h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetReservationByOrderID failed: %v", err)
	}
	if reservation.Status != models.ReservationStatusConfirmed {
		t.Errorf("reservation status = %s, want %s", reservation.Status, models.ReservationStatusConfirmed)
	}

	// --- Shipping assertions ---
	shipment, err := h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetShipmentByOrderID failed: %v", err)
	}
	if shipment.Status != models.ShipmentStatusScheduled {
		t.Errorf("shipment status = %s, want %s", shipment.Status, models.ShipmentStatusScheduled)
	}
	if shipment.TrackingNumber == "" {
		t.Error("shipment TrackingNumber is empty; expected a tracking number")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "COMPLETED" {
		t.Errorf("saga monitor status = %s, want COMPLETED", sagaState.Status)
	}
}

// ---------------------------------------------------------------------------
// Test 2: Payment Failure Saga
// ---------------------------------------------------------------------------
// When a customer has insufficient funds, the saga should fail early:
// - The order is cancelled
// - The payment is recorded as FAILED (not COMPLETED, not REFUNDED)
// - The customer's balance is untouched (deduction never happened)
// - No inventory reservation is created
// - No shipment is created
//
// This is the simplest failure case because no compensation is needed -
// the saga fails at the first step, so there's nothing downstream to undo.

func TestPaymentFailureSaga(t *testing.T) {
	h := newSyncTestHarness(t)

	// CUST-002 has only $50; the order will cost $999.99.
	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-002")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-001")

	items := []models.OrderItem{
		{ProductID: "PROD-001", Quantity: 1, Price: 999.99},
	}

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-002", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCancelled {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCancelled)
	}

	// --- Payment assertions ---
	// A payment record is created (as PENDING) but then marked FAILED
	// when the balance check fails. It should never reach COMPLETED or REFUNDED.
	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusFailed {
		t.Errorf("payment status = %s, want %s", pay.Status, models.PaymentStatusFailed)
	}

	// Customer balance must be unchanged - the deduction was never executed.
	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-002")
	if !almostEqual(customerAfter.Balance, customerBefore.Balance) {
		t.Errorf("customer balance = %.2f, want %.2f (unchanged)",
			customerAfter.Balance, customerBefore.Balance)
	}

	// --- Inventory assertions ---
	// The saga never reached the inventory step, so stock is untouched.
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-001")
	if productAfter.Quantity != productBefore.Quantity {
		t.Errorf("product PROD-001 quantity = %d, want %d (unchanged)",
			productAfter.Quantity, productBefore.Quantity)
	}

	// No reservation should exist for this order.
	_, err = h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no reservation for failed-payment order, but found one")
	}

	// --- Shipping assertions ---
	// No shipment should exist for this order.
	_, err = h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no shipment for failed-payment order, but found one")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "CANCELLED" {
		t.Errorf("saga monitor status = %s, want CANCELLED", sagaState.Status)
	}
}

// ---------------------------------------------------------------------------
// Test 3: Inventory Failure Saga
// ---------------------------------------------------------------------------
// When payment succeeds but inventory reservation fails (out of stock),
// the saga must compensate by refunding the payment:
// - The order is cancelled
// - The payment progresses PENDING → COMPLETED → REFUNDED
// - The customer's balance is fully restored
// - No inventory reservation exists (the reserve operation failed)
// - No shipment is created
//
// This tests the compensation chain: order.cancelled → Payment service refund.

func TestInventoryFailureSaga(t *testing.T) {
	h := newSyncTestHarness(t)

	// CUST-001 has $1000; PROD-004 (mechanical keyboard) has 0 stock.
	// Payment for $149.99 will succeed, but inventory reservation will fail.
	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-001")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-004")

	items := []models.OrderItem{
		{ProductID: "PROD-004", Quantity: 1, Price: 149.99},
	}

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-001", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCancelled {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCancelled)
	}

	// --- Payment assertions ---
	// The payment was successfully completed, then refunded as compensation
	// when the inventory step failed and the saga was rolled back.
	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusRefunded {
		t.Errorf("payment status = %s, want %s (should have been refunded)",
			pay.Status, models.PaymentStatusRefunded)
	}

	// Customer balance must be fully restored: the payment was deducted,
	// then credited back. Net effect on the balance should be zero.
	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-001")
	if !almostEqual(customerAfter.Balance, customerBefore.Balance) {
		t.Errorf("customer balance = %.2f, want %.2f (restored after refund)",
			customerAfter.Balance, customerBefore.Balance)
	}

	// --- Inventory assertions ---
	// PROD-004 had 0 stock and reservation failed, so quantity is still 0.
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-004")
	if productAfter.Quantity != productBefore.Quantity {
		t.Errorf("product PROD-004 quantity = %d, want %d (unchanged)",
			productAfter.Quantity, productBefore.Quantity)
	}

	// No reservation should exist - the reserve operation itself failed.
	_, err = h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no reservation for out-of-stock order, but found one")
	}

	// --- Shipping assertions ---
	// The saga never reached shipping.
	_, err = h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no shipment for failed-inventory order, but found one")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "CANCELLED" {
		t.Errorf("saga monitor status = %s, want CANCELLED", sagaState.Status)
	}
}

// ---------------------------------------------------------------------------
// Test 4: Shipping Failure Saga (Full Compensation)
// ---------------------------------------------------------------------------
// This is the most complex failure scenario. Payment succeeds, inventory is
// reserved, but shipping fails. The saga must compensate BOTH previous steps:
// - The order is cancelled
// - The payment is refunded (balance restored)
// - The inventory reservation is released (stock restored)
// - No shipment exists (the failure occurs before shipment creation)
//
// This tests the full compensation chain:
//   order.cancelled → Payment service refund + Inventory service release

func TestShippingFailureSaga(t *testing.T) {
	h := newSyncTestHarness(t)

	// Enable the simulated shipping failure.
	// The Shipping service will publish "shipment.schedule_failed" without
	// creating a shipment, which triggers the full compensation chain.
	h.shippingSvc.SetSimulateFailure(true)

	// CUST-003 has $5000; PROD-002 (wireless mouse) has 100 units.
	// Payment and inventory reservation will both succeed before shipping fails.
	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-003")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-002")

	items := []models.OrderItem{
		{ProductID: "PROD-002", Quantity: 5, Price: 29.99},
	}

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-003", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCancelled {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCancelled)
	}

	// --- Payment assertions ---
	// Payment was completed, then refunded as compensation.
	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusRefunded {
		t.Errorf("payment status = %s, want %s (should have been refunded)",
			pay.Status, models.PaymentStatusRefunded)
	}

	// Customer balance must be fully restored.
	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-003")
	if !almostEqual(customerAfter.Balance, customerBefore.Balance) {
		t.Errorf("customer balance = %.2f, want %.2f (restored after refund)",
			customerAfter.Balance, customerBefore.Balance)
	}

	// --- Inventory assertions ---
	// The reservation was created and confirmed, then released as compensation.
	// Stock should be fully restored to its original level.
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-002")
	if productAfter.Quantity != productBefore.Quantity {
		t.Errorf("product PROD-002 quantity = %d, want %d (restored after release)",
			productAfter.Quantity, productBefore.Quantity)
	}

	// The reservation should exist but be in RELEASED status.
	// This proves the compensation ran: the reservation wasn't deleted,
	// it was explicitly released - maintaining an audit trail.
	reservation, err := h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetReservationByOrderID failed: %v", err)
	}
	if reservation.Status != models.ReservationStatusReleased {
		t.Errorf("reservation status = %s, want %s",
			reservation.Status, models.ReservationStatusReleased)
	}

	// --- Shipping assertions ---
	// No shipment should exist. The simulated failure occurs BEFORE the
	// shipment record is created, so there's nothing for the Shipping
	// service to cancel when it receives "order.cancelled".
	// This validates that compensation is smart: each service only
	// undoes work it actually completed.
	_, err = h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no shipment for failed-shipping order, but found one")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "CANCELLED" {
		t.Errorf("saga monitor status = %s, want CANCELLED", sagaState.Status)
	}
}

// ============================================================================
// Part B: Asynchronous delivery tests
// ============================================================================
//
// These tests use asynchronous event delivery, where each subscriber receives
// events in its own goroutine. This mirrors real production behavior with a
// message broker like Kafka or NATS, where:
//   - Events are delivered concurrently to all subscribers.
//   - There are no ordering guarantees between subscribers.
//   - Multiple events may be in flight simultaneously.
//   - Race conditions between concurrent handlers are possible and expected.
//
// Async tests validate concurrency correctness: do mutexes, atomic operations,
// and idempotency guards work correctly under concurrent event processing?
// Do compensating transactions complete reliably when events race?
//
// The key structural difference from sync tests is the waiting strategy:
//
//   Sync:   CreateOrder() returns → saga is fully complete → assert immediately
//   Async:  CreateOrder() returns → saga runs in background → wait for terminal
//           event → poll for compensation to settle → assert
//
// We use two layers of waiting:
//   1. eventBus.WaitForSaga() blocks until a terminal event (order.completed or
//      order.cancelled) is published. This tells us the saga reached a decision.
//   2. waitForCondition() polls for specific downstream effects (refunds, stock
//      restoration) that happen AFTER the terminal event. This is necessary
//      because order.cancelled triggers compensation events that are still
//      in flight when WaitForSaga returns.
//
// These tests also validate a subtle but important property: subscriber
// registration order is irrelevant in async mode. The newAsyncTestHarness
// deliberately creates services in a different order than the sync harness,
// proving that the saga reaches the same correct outcome regardless.

// ---------------------------------------------------------------------------
// Async Test 1: Successful Order Saga (Happy Path)
// ---------------------------------------------------------------------------
// Validates that the happy path works correctly under concurrent event delivery.
//
// The async happy path is interesting because it exercises the concurrent fan-out
// of the order.created event to three services simultaneously. All three services
// process order.created in parallel:
//   - Payment processes the payment and publishes payment.completed
//   - Inventory caches order items (for later use)
//   - Shipping caches customer data (for later use)
//
// Then payment.completed triggers inventory reservation, and inventory.reserved
// triggers shipping scheduling. The Order service receives success events from
// all three steps concurrently and must correctly determine when all three have
// succeeded using its atomic RecordStepResult method.

func TestAsyncSuccessfulOrderSaga(t *testing.T) {
	h := newAsyncTestHarness(t)

	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-003")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-002")

	items := []models.OrderItem{
		{ProductID: "PROD-002", Quantity: 3, Price: 29.99},
	}
	totalAmount := 3 * 29.99

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-003", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	// In async mode, CreateOrder returns immediately after publishing
	// order.created. The saga runs in the background. We must wait for it.
	if err = h.eventBus.WaitForSaga(h.ctx, sagaTimeout); err != nil {
		t.Fatalf("Saga did not complete: %v", err)
	}

	// For the happy path, there's no compensation after the terminal event.
	// The WaitForSaga 50ms buffer is sufficient. But we still use
	// waitForCondition to be robust against scheduling delays.
	waitForCondition(t, compensationTimeout, func() bool {
		o, err := h.orderSvc.GetOrder(newOrder.ID)
		return err == nil && o.Status == models.OrderStatusCompleted
	}, "order to reach COMPLETED status")

	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCompleted {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCompleted)
	}
	if finalOrder.PaymentID == "" {
		t.Error("order PaymentID is empty; expected a payment reference")
	}
	if finalOrder.ReservationID == "" {
		t.Error("order ReservationID is empty; expected a reservation reference")
	}
	if finalOrder.ShipmentID == "" {
		t.Error("order ShipmentID is empty; expected a shipment reference")
	}

	// --- Payment assertions ---
	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusCompleted {
		t.Errorf("payment status = %s, want %s", pay.Status, models.PaymentStatusCompleted)
	}

	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-003")
	expectedBalance := customerBefore.Balance - totalAmount
	if !almostEqual(customerAfter.Balance, expectedBalance) {
		t.Errorf("customer balance = %.2f, want %.2f", customerAfter.Balance, expectedBalance)
	}

	// --- Inventory assertions ---
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-002")
	expectedQty := productBefore.Quantity - 3
	if productAfter.Quantity != expectedQty {
		t.Errorf("product PROD-002 quantity = %d, want %d", productAfter.Quantity, expectedQty)
	}

	reservation, err := h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetReservationByOrderID failed: %v", err)
	}
	if reservation.Status != models.ReservationStatusConfirmed {
		t.Errorf("reservation status = %s, want %s", reservation.Status, models.ReservationStatusConfirmed)
	}

	// --- Shipping assertions ---
	shipment, err := h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetShipmentByOrderID failed: %v", err)
	}
	if shipment.Status != models.ShipmentStatusScheduled {
		t.Errorf("shipment status = %s, want %s", shipment.Status, models.ShipmentStatusScheduled)
	}
	if shipment.TrackingNumber == "" {
		t.Error("shipment TrackingNumber is empty; expected a tracking number")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "COMPLETED" {
		t.Errorf("saga monitor status = %s, want COMPLETED", sagaState.Status)
	}
}

// ---------------------------------------------------------------------------
// Async Test 2: Payment Failure Saga
// ---------------------------------------------------------------------------
// When payment fails, the saga is short-circuited. No downstream work was
// performed, so no compensation is needed. The interesting async aspect here
// is that the Inventory and Shipping services' order.created handlers may
// still be running (caching data) when payment.failed arrives at the Order
// service. The Order service must handle this correctly: recording the payment
// failure and cancelling the saga regardless of what other services are doing.
//
// The order.cancelled event then fans out to all three services concurrently.
// Each service independently checks its local state and finds nothing to
// compensate (no completed payment, no reservation, no shipment), so all
// three compensation handlers are effectively no-ops.

func TestAsyncPaymentFailureSaga(t *testing.T) {
	h := newAsyncTestHarness(t)

	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-002")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-001")

	items := []models.OrderItem{
		{ProductID: "PROD-001", Quantity: 1, Price: 999.99},
	}

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-002", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	// Wait for the terminal event (order.cancelled in this case).
	if err = h.eventBus.WaitForSaga(h.ctx, sagaTimeout); err != nil {
		t.Fatalf("Saga did not complete: %v", err)
	}

	// No compensation is needed for payment failure, but we poll for the
	// order status to be fully settled just in case of scheduling delays.
	waitForCondition(t, compensationTimeout, func() bool {
		o, err := h.orderSvc.GetOrder(newOrder.ID)
		return err == nil && o.Status == models.OrderStatusCancelled
	}, "order to reach CANCELLED status")

	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCancelled {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCancelled)
	}

	// --- Payment assertions ---
	// Poll for the payment record to exist, since it's created asynchronously.
	waitForCondition(t, compensationTimeout, func() bool {
		_, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
		return err == nil
	}, "payment record to be created")

	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusFailed {
		t.Errorf("payment status = %s, want %s", pay.Status, models.PaymentStatusFailed)
	}

	// Customer balance must be unchanged.
	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-002")
	if !almostEqual(customerAfter.Balance, customerBefore.Balance) {
		t.Errorf("customer balance = %.2f, want %.2f (unchanged)",
			customerAfter.Balance, customerBefore.Balance)
	}

	// --- Inventory assertions ---
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-001")
	if productAfter.Quantity != productBefore.Quantity {
		t.Errorf("product PROD-001 quantity = %d, want %d (unchanged)",
			productAfter.Quantity, productBefore.Quantity)
	}

	_, err = h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no reservation for failed-payment order, but found one")
	}

	// --- Shipping assertions ---
	_, err = h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no shipment for failed-payment order, but found one")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "CANCELLED" {
		t.Errorf("saga monitor status = %s, want CANCELLED", sagaState.Status)
	}
}

// ---------------------------------------------------------------------------
// Async Test 3: Inventory Failure Saga
// ---------------------------------------------------------------------------
// This test exercises the compensation chain under concurrent delivery.
// The sequence is: payment succeeds → inventory fails → order cancelled →
// payment refunded. The critical async behavior here is:
//
//   1. payment.completed and inventory.reserve_failed may arrive at the Order
//      service concurrently. The RecordStepResult mutex ensures exactly one
//      of them triggers the terminal SagaActionCancel.
//
//   2. After order.cancelled is published, the Payment service's refund handler
//      runs concurrently with the Inventory and Shipping services' no-op
//      compensation handlers. The refund must complete correctly despite this
//      concurrency.
//
// We poll specifically for the payment status to reach REFUNDED, which proves
// the full compensation chain completed.

func TestAsyncInventoryFailureSaga(t *testing.T) {
	h := newAsyncTestHarness(t)

	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-001")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-004")

	items := []models.OrderItem{
		{ProductID: "PROD-004", Quantity: 1, Price: 149.99},
	}

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-001", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	// Wait for the terminal event.
	if err = h.eventBus.WaitForSaga(h.ctx, sagaTimeout); err != nil {
		t.Fatalf("Saga did not complete: %v", err)
	}

	// The terminal event is order.cancelled. After it's published, the Payment
	// service receives it and issues a refund asynchronously. We must poll
	// for the refund to complete before asserting on the payment status and
	// the customer's balance.
	waitForCondition(t, compensationTimeout, func() bool {
		pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
		return err == nil && pay.Status == models.PaymentStatusRefunded
	}, "payment to reach REFUNDED status (compensation)")

	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCancelled {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCancelled)
	}

	// --- Payment assertions ---
	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusRefunded {
		t.Errorf("payment status = %s, want %s (should have been refunded)",
			pay.Status, models.PaymentStatusRefunded)
	}

	// Customer balance must be fully restored after the refund.
	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-001")
	if !almostEqual(customerAfter.Balance, customerBefore.Balance) {
		t.Errorf("customer balance = %.2f, want %.2f (restored after refund)",
			customerAfter.Balance, customerBefore.Balance)
	}

	// --- Inventory assertions ---
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-004")
	if productAfter.Quantity != productBefore.Quantity {
		t.Errorf("product PROD-004 quantity = %d, want %d (unchanged)",
			productAfter.Quantity, productBefore.Quantity)
	}

	_, err = h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no reservation for out-of-stock order, but found one")
	}

	// --- Shipping assertions ---
	_, err = h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no shipment for failed-inventory order, but found one")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "CANCELLED" {
		t.Errorf("saga monitor status = %s, want CANCELLED", sagaState.Status)
	}
}

// ---------------------------------------------------------------------------
// Async Test 4: Shipping Failure Saga (Full Compensation)
// ---------------------------------------------------------------------------
// This is the most complex async scenario because it exercises the WIDEST
// compensation chain: both payment and inventory must be compensated.
//
// The async dimension adds several interesting concurrency scenarios:
//
//   1. Three success/failure events race to reach the Order service:
//      payment.completed, inventory.reserved, and shipment.schedule_failed.
//      These may arrive in any order. The Order service's RecordStepResult
//      method uses a mutex to evaluate all three atomically. Whichever
//      goroutine records the shipping failure will observe that a step has
//      failed and receive SagaActionCancel. The others will find the order
//      already in CANCELLED status and receive SagaActionNone.
//
//   2. After order.cancelled fans out, three compensation handlers run
//      concurrently:
//        - Payment service: finds a completed payment → issues refund
//        - Inventory service: finds a confirmed reservation → releases stock
//        - Shipping service: finds no shipment → no-op
//      The refund and release happen in parallel, and both must complete
//      correctly for the system to reach a consistent state.
//
// We poll for BOTH the payment refund AND the inventory release to complete,
// since both are independent compensation paths that run concurrently.

func TestAsyncShippingFailureSaga(t *testing.T) {
	h := newAsyncTestHarness(t)

	h.shippingSvc.SetSimulateFailure(true)

	customerBefore, _ := h.paymentRepo.GetCustomer("CUST-003")
	productBefore, _ := h.inventoryRepo.GetProduct("PROD-002")

	items := []models.OrderItem{
		{ProductID: "PROD-002", Quantity: 5, Price: 29.99},
	}

	newOrder, err := h.orderSvc.CreateOrder(h.ctx, "CUST-003", items)
	if err != nil {
		t.Fatalf("CreateOrder failed: %v", err)
	}

	// Wait for the terminal event.
	if err = h.eventBus.WaitForSaga(h.ctx, sagaTimeout); err != nil {
		t.Fatalf("Saga did not complete: %v", err)
	}

	// Two independent compensation paths run in parallel after order.cancelled:
	//   Path 1: Payment service refunds the payment
	//   Path 2: Inventory service releases the reservation
	//
	// We must wait for BOTH to complete. We poll for the payment refund first
	// (since it's the most financially critical), then for the inventory release.
	waitForCondition(t, compensationTimeout, func() bool {
		pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
		return err == nil && pay.Status == models.PaymentStatusRefunded
	}, "payment to reach REFUNDED status (compensation)")

	waitForCondition(t, compensationTimeout, func() bool {
		res, err := h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
		return err == nil && res.Status == models.ReservationStatusReleased
	}, "reservation to reach RELEASED status (compensation)")

	finalOrder, err := h.orderSvc.GetOrder(newOrder.ID)
	if err != nil {
		t.Fatalf("GetOrder failed: %v", err)
	}

	// --- Order assertions ---
	if finalOrder.Status != models.OrderStatusCancelled {
		t.Errorf("order status = %s, want %s", finalOrder.Status, models.OrderStatusCancelled)
	}

	// --- Payment assertions ---
	pay, err := h.paymentRepo.GetPaymentByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetPaymentByOrderID failed: %v", err)
	}
	if pay.Status != models.PaymentStatusRefunded {
		t.Errorf("payment status = %s, want %s (should have been refunded)",
			pay.Status, models.PaymentStatusRefunded)
	}

	// Customer balance must be fully restored: deducted during payment,
	// credited back during refund. Net effect = zero.
	customerAfter, _ := h.paymentRepo.GetCustomer("CUST-003")
	if !almostEqual(customerAfter.Balance, customerBefore.Balance) {
		t.Errorf("customer balance = %.2f, want %.2f (restored after refund)",
			customerAfter.Balance, customerBefore.Balance)
	}

	// --- Inventory assertions ---
	// Stock must be fully restored: deducted during reservation, credited
	// back during release. Net effect = zero.
	productAfter, _ := h.inventoryRepo.GetProduct("PROD-002")
	if productAfter.Quantity != productBefore.Quantity {
		t.Errorf("product PROD-002 quantity = %d, want %d (restored after release)",
			productAfter.Quantity, productBefore.Quantity)
	}

	// The reservation should exist but be in RELEASED status, providing
	// an audit trail that compensation ran.
	reservation, err := h.inventoryRepo.GetReservationByOrderID(newOrder.ID)
	if err != nil {
		t.Fatalf("GetReservationByOrderID failed: %v", err)
	}
	if reservation.Status != models.ReservationStatusReleased {
		t.Errorf("reservation status = %s, want %s",
			reservation.Status, models.ReservationStatusReleased)
	}

	// --- Shipping assertions ---
	_, err = h.shippingRepo.GetShipmentByOrderID(newOrder.ID)
	if err == nil {
		t.Error("expected no shipment for failed-shipping order, but found one")
	}

	// --- Monitor assertions ---
	sagaState, err := h.monitor.GetSagaState(newOrder.ID)
	if err != nil {
		t.Fatalf("GetSagaState failed: %v", err)
	}
	if sagaState.Status != "CANCELLED" {
		t.Errorf("saga monitor status = %s, want CANCELLED", sagaState.Status)
	}
}

// ============================================================================
// Part C: Async stress tests
// ============================================================================
//
// The tests above run a single saga per test. In production, multiple sagas
// execute concurrently, sharing the same event bus, repositories, and services.
// The tests below validate that the system handles concurrent sagas correctly
// without cross-contamination (e.g., one saga's refund affecting another
// saga's payment, or correlation IDs getting mixed up).

// ---------------------------------------------------------------------------
// Async Test 5: Concurrent Sagas
// ---------------------------------------------------------------------------
// Launches multiple sagas simultaneously and verifies that each one reaches
// the correct terminal state independently. This validates:
//   - Correlation IDs correctly isolate saga instances
//   - Repository-level mutexes don't cause deadlocks under concurrency
//   - Idempotency guards don't interfere across saga instances
//   - The event bus delivers events to all subscribers for all sagas

func TestAsyncConcurrentSagas(t *testing.T) {
	h := newAsyncTestHarness(t)

	// We'll launch three sagas concurrently:
	//   Saga A: Happy path (CUST-003 buys PROD-002) → COMPLETED
	//   Saga B: Payment failure (CUST-002 buys expensive item) → CANCELLED
	//   Saga C: Inventory failure (CUST-001 buys out-of-stock PROD-004) → CANCELLED

	type sagaExpectation struct {
		customerID     string
		items          []models.OrderItem
		expectedStatus models.OrderStatus
	}

	scenarios := []sagaExpectation{
		{
			customerID: "CUST-003",
			items: []models.OrderItem{
				{ProductID: "PROD-002", Quantity: 1, Price: 29.99},
			},
			expectedStatus: models.OrderStatusCompleted,
		},
		{
			customerID: "CUST-002",
			items: []models.OrderItem{
				{ProductID: "PROD-001", Quantity: 10, Price: 999.99}, // Exceeds CUST-002's $50
			},
			expectedStatus: models.OrderStatusCancelled,
		},
		{
			customerID: "CUST-001",
			items: []models.OrderItem{
				{ProductID: "PROD-004", Quantity: 1, Price: 149.99}, // 0 stock
			},
			expectedStatus: models.OrderStatusCancelled,
		},
	}

	// Launch all sagas concurrently.
	orderIDs := make([]string, len(scenarios))
	for i, sc := range scenarios {
		newOrder, err := h.orderSvc.CreateOrder(h.ctx, sc.customerID, sc.items)
		if err != nil {
			t.Fatalf("CreateOrder[%d] failed: %v", i, err)
		}
		orderIDs[i] = newOrder.ID
	}

	// Wait for ALL sagas to reach terminal states by polling each one.
	// We can't use WaitForSaga here because it only signals once per terminal
	// event, and we have multiple sagas. Instead, we poll each order's status.
	for i, sc := range scenarios {
		expectedStatus := sc.expectedStatus
		orderID := orderIDs[i]

		waitForCondition(t, sagaTimeout, func() bool {
			o, err := h.orderSvc.GetOrder(orderID)
			if err != nil {
				return false
			}
			return o.Status == expectedStatus
		}, fmt.Sprintf("saga %d (order %s) to reach %s", i, orderID, expectedStatus))
	}

	// Allow compensation events to settle for the cancelled sagas.
	// Saga C (inventory failure) needs its payment refunded.
	waitForCondition(t, compensationTimeout, func() bool {
		pay, err := h.paymentRepo.GetPaymentByOrderID(orderIDs[2])
		return err == nil && pay.Status == models.PaymentStatusRefunded
	}, "saga C payment refund to complete")

	// --- Verify each saga reached the correct state ---

	// Saga A: should be COMPLETED.
	orderA, _ := h.orderSvc.GetOrder(orderIDs[0])
	if orderA.Status != models.OrderStatusCompleted {
		t.Errorf("saga A: order status = %s, want %s", orderA.Status, models.OrderStatusCompleted)
	}
	if orderA.PaymentID == "" || orderA.ReservationID == "" || orderA.ShipmentID == "" {
		t.Error("saga A: expected all resource references to be populated")
	}

	// Saga B: should be CANCELLED (payment failure, no compensation needed).
	orderB, _ := h.orderSvc.GetOrder(orderIDs[1])
	if orderB.Status != models.OrderStatusCancelled {
		t.Errorf("saga B: order status = %s, want %s", orderB.Status, models.OrderStatusCancelled)
	}

	// Saga C: should be CANCELLED with payment refunded.
	orderC, _ := h.orderSvc.GetOrder(orderIDs[2])
	if orderC.Status != models.OrderStatusCancelled {
		t.Errorf("saga C: order status = %s, want %s", orderC.Status, models.OrderStatusCancelled)
	}

	// Verify the monitor tracked all three sagas independently.
	metrics := h.monitor.GetMetrics()
	if metrics.TotalSagas != 3 {
		t.Errorf("total sagas = %d, want 3", metrics.TotalSagas)
	}
	if metrics.CompletedSagas != 1 {
		t.Errorf("completed sagas = %d, want 1", metrics.CompletedSagas)
	}
	if metrics.CancelledSagas != 2 {
		t.Errorf("cancelled sagas = %d, want 2", metrics.CancelledSagas)
	}
}
