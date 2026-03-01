package order

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"

	"saga-choreography/pkg/models"
)

// StepName identifies which saga step is reporting a result.
type StepName string

const (
	StepPayment   StepName = "payment"
	StepInventory StepName = "inventory"
	StepShipping  StepName = "shipping"
)

// StepRefs carries optional resource references to store alongside a step result.
// When a downstream service completes work (e.g., creates a payment or reservation),
// it includes the resource ID in its success event. The Order service passes these
// through StepRefs so they're stored atomically with the step result update.
type StepRefs struct {
	PaymentID      string
	ReservationID  string
	ShipmentID     string
	TrackingNumber string
}

// Repository provides storage for orders.
// In production, this would be backed by a database.
//
// All methods accept a context.Context as their first parameter. While our
// in-memory implementation doesn't use the context, this signature matches
// what a production database-backed repository would require (for query
// timeouts, cancellation, and tracing). Teaching the correct habit from
// the start prevents a painful refactor later.
type Repository struct {
	mu     sync.RWMutex
	orders map[string]*models.Order
}

// NewRepository creates a new order repository.
func NewRepository() *Repository {
	return &Repository{
		orders: make(map[string]*models.Order),
	}
}

// Create stores a new order. All step results are initialized to StepPending.
func (r *Repository) Create(ctx context.Context, order *models.Order) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.orders[order.ID]; exists {
		return fmt.Errorf("order %s already exists", order.ID)
	}

	order.CreatedAt = time.Now()
	order.UpdatedAt = time.Now()
	order.PaymentStep = models.StepPending
	order.InventoryStep = models.StepPending
	order.ShippingStep = models.StepPending
	r.orders[order.ID] = order
	return nil
}

// Get retrieves an order by ID.
func (r *Repository) Get(ctx context.Context, id string) (*models.Order, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	order, exists := r.orders[id]
	if !exists {
		return nil, fmt.Errorf("order %s not found", id)
	}

	// Return a copy to avoid race conditions.
	// Without this, a caller could modify the order through the returned pointer
	// while another goroutine is reading it, causing data races.
	//
	// Go 1.26's new(expr) syntax lets us write new(*order) instead of the old
	// pattern of declaring a variable and taking its address. The expression
	// *order dereferences the pointer to produce a value, and new(...) allocates
	// a fresh variable initialized to that value, returning a pointer to it.
	orderCopy := new(*order)
	orderCopy.Items = make([]models.OrderItem, len(order.Items))
	copy(orderCopy.Items, order.Items)
	return orderCopy, nil
}

// RecordStepResult atomically records a step's outcome, stores any resource
// references, and evaluates whether the saga has reached a terminal state.
//
// This is the central method that makes out-of-order event handling safe.
// The entire read-modify-evaluate cycle happens under a single mutex lock,
// which guarantees that even if two goroutines deliver events simultaneously
// (e.g., payment.completed and inventory.reserve_failed racing), exactly one
// of them will observe the transition to a terminal state and receive a
// non-None SagaAction. The other will find the order already in a terminal
// status and receive SagaActionNone.
func (r *Repository) RecordStepResult(ctx context.Context, orderID string, step StepName, result models.StepResult, refs StepRefs) (models.SagaAction, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	order, exists := r.orders[orderID]
	if !exists {
		return models.SagaActionNone, fmt.Errorf("order %s not found", orderID)
	}

	// Terminal orders accept no further updates.
	// This is the guard that prevents double-publishing of order.completed
	// or order.cancelled when two events race to complete/cancel the saga.
	if order.Status == models.OrderStatusCompleted || order.Status == models.OrderStatusCancelled {
		return models.SagaActionNone, nil
	}

	// 1. Record the step result
	switch step {
	case StepPayment:
		order.PaymentStep = result
	case StepInventory:
		order.InventoryStep = result
	case StepShipping:
		order.ShippingStep = result
	}

	// 2. Store any resource references
	if refs.PaymentID != "" {
		order.PaymentID = refs.PaymentID
	}
	if refs.ReservationID != "" {
		order.ReservationID = refs.ReservationID
	}
	if refs.ShipmentID != "" {
		order.ShipmentID = refs.ShipmentID
	}
	if refs.TrackingNumber != "" {
		order.TrackingNumber = refs.TrackingNumber
	}

	order.UpdatedAt = time.Now()

	// 3. Evaluate the saga state based on all step results
	return r.evaluateSaga(order), nil
}

// evaluateSaga examines the three step results and determines whether the saga
// has reached a terminal state. It also updates the order's Status field for
// observability.
//
// The evaluation logic is simple and deliberate:
//   - Any failure - cancel immediately. We don't wait for other steps to report
//     because a failure is definitive: the saga cannot succeed. The order.cancelled
//     event will reach all services, and each one independently checks its local
//     state to decide whether compensation is needed.
//   - All three succeeded - complete. This is the only path to success.
//   - Otherwise - still in progress. Update the status to reflect the most
//     advanced successful step for observability, but take no terminal action.
func (r *Repository) evaluateSaga(order *models.Order) models.SagaAction {
	// Check for any failure - this takes priority over everything else.
	// A single failed step is enough to doom the saga, regardless of what
	// other steps have reported (or haven't reported yet).
	if order.PaymentStep == models.StepFailed {
		order.Status = models.OrderStatusCancelled
		return models.SagaActionCancel
	}
	if order.InventoryStep == models.StepFailed {
		order.Status = models.OrderStatusCancelled
		return models.SagaActionCancel
	}
	if order.ShippingStep == models.StepFailed {
		order.Status = models.OrderStatusCancelled
		return models.SagaActionCancel
	}

	// Check for full success - all three steps must have reported success.
	if order.PaymentStep == models.StepSucceeded &&
		order.InventoryStep == models.StepSucceeded &&
		order.ShippingStep == models.StepSucceeded {
		order.Status = models.OrderStatusCompleted
		return models.SagaActionComplete
	}

	// Still in progress. Derive the most descriptive intermediate status
	// from the step results so that queries and dashboards can show how
	// far the saga has progressed.
	switch {
	case order.PaymentStep == models.StepSucceeded && order.InventoryStep == models.StepSucceeded:
		order.Status = models.OrderStatusInventoryReserved
	case order.PaymentStep == models.StepSucceeded:
		order.Status = models.OrderStatusPaymentCompleted
	default:
		order.Status = models.OrderStatusPending
	}

	return models.SagaActionNone
}

// MarkStepCompensated records that a step's work has been undone.
// This is called when compensation confirmation events arrive (payment.refunded,
// inventory.released). It's purely informational and doesn't change the saga
// outcome - the order is already CANCELLED by this point.
func (r *Repository) MarkStepCompensated(ctx context.Context, orderID string, step StepName) {
	r.mu.Lock()
	defer r.mu.Unlock()

	order, exists := r.orders[orderID]
	if !exists {
		return
	}

	switch step {
	case StepPayment:
		order.PaymentStep = models.StepCompensated
	case StepInventory:
		order.InventoryStep = models.StepCompensated
	}

	order.UpdatedAt = time.Now()
}

// All returns an iterator over all orders.
func (r *Repository) All() iter.Seq[*models.Order] {
	return func(yield func(*models.Order) bool) {
		r.mu.RLock()
		defer r.mu.RUnlock()

		for _, order := range r.orders {
			// new(*order) creates a safe copy using Go 1.26's new(expr) syntax.
			if !yield(new(*order)) {
				return
			}
		}
	}
}
