package models

import (
	"time"
)

// OrderStatus represents the current state of an order in the saga.
// These statuses form a state machine that tracks how far the saga has progressed
// and whether it completed successfully or was rolled back.
type OrderStatus string

const (
	OrderStatusPending           OrderStatus = "PENDING"
	OrderStatusPaymentCompleted  OrderStatus = "PAYMENT_COMPLETED"
	OrderStatusPaymentFailed     OrderStatus = "PAYMENT_FAILED"
	OrderStatusInventoryReserved OrderStatus = "INVENTORY_RESERVED"
	OrderStatusInventoryFailed   OrderStatus = "INVENTORY_FAILED"
	OrderStatusShippingFailed    OrderStatus = "SHIPPING_FAILED"
	OrderStatusCompleted         OrderStatus = "COMPLETED"
	OrderStatusCancelled         OrderStatus = "CANCELLED"
)

// StepResult tracks the outcome of an individual saga step.
// Each step in the saga (payment, inventory, shipping) progresses independently
// through these states. This decoupling is what allows the Order service to
// handle events arriving in any order.
type StepResult string

const (
	// StepPending means the step has not yet reported a result.
	StepPending StepResult = "PENDING"

	// StepSucceeded means the step completed its work successfully.
	StepSucceeded StepResult = "SUCCEEDED"

	// StepFailed means the step could not complete its work.
	StepFailed StepResult = "FAILED"

	// StepCompensated means the step's work was successfully undone.
	// This is set when a compensation confirmation event arrives (e.g.,
	// payment.refunded or inventory.released). It's purely informational
	// and doesn't affect saga evaluation.
	StepCompensated StepResult = "COMPENSATED"
)

// SagaAction tells the Order service what to do after recording a step result.
// The repository's RecordStepResult method evaluates all three step results
// atomically (under a mutex) and returns one of these actions. This ensures
// that even if two events race to update different steps, exactly one of them
// will trigger the terminal action.
type SagaAction int

const (
	// SagaActionNone means the saga is still in progress - more step results
	// are needed before a terminal decision can be made.
	SagaActionNone SagaAction = iota

	// SagaActionComplete means all three steps succeeded. The service should
	// publish order.completed.
	SagaActionComplete

	// SagaActionCancel means at least one step failed. The service should
	// publish order.cancelled to trigger compensating transactions.
	SagaActionCancel
)

// Order represents a customer order.
type Order struct {
	ID          string      `json:"id"`
	CustomerID  string      `json:"customer_id"`
	Items       []OrderItem `json:"items"`
	TotalAmount float64     `json:"total_amount"`
	Status      OrderStatus `json:"status"`
	CreatedAt   time.Time   `json:"created_at"`
	UpdatedAt   time.Time   `json:"updated_at"`

	// Step-level tracking for out-of-order event handling.
	// Each field records the outcome of one saga step independently.
	// The overall OrderStatus is derived from these three fields by the
	// repository's evaluateSaga method, rather than being set directly
	// by individual event handlers.
	PaymentStep   StepResult `json:"payment_step"`
	InventoryStep StepResult `json:"inventory_step"`
	ShippingStep  StepResult `json:"shipping_step"`

	// Saga-related fields for tracking compensation.
	// These IDs let the Order service know which resources were created
	// downstream, which is essential for understanding what needs to be
	// undone if the saga fails.
	PaymentID      string `json:"payment_id,omitempty"`
	ReservationID  string `json:"reservation_id,omitempty"`
	ShipmentID     string `json:"shipment_id,omitempty"`
	TrackingNumber string `json:"tracking_number,omitempty"`
}

// OrderItem represents a single item in an order.
type OrderItem struct {
	ProductID string  `json:"product_id"`
	Quantity  int     `json:"quantity"`
	Price     float64 `json:"price"`
}

// Payment represents a payment transaction.
type Payment struct {
	ID         string        `json:"id"`
	OrderID    string        `json:"order_id"`
	CustomerID string        `json:"customer_id"`
	Amount     float64       `json:"amount"`
	Status     PaymentStatus `json:"status"`
	CreatedAt  time.Time     `json:"created_at"`
}

// PaymentStatus represents the state of a payment.
type PaymentStatus string

const (
	PaymentStatusPending   PaymentStatus = "PENDING"
	PaymentStatusCompleted PaymentStatus = "COMPLETED"
	PaymentStatusFailed    PaymentStatus = "FAILED"
	PaymentStatusRefunded  PaymentStatus = "REFUNDED"
)

// InventoryReservation represents a reservation of inventory items.
type InventoryReservation struct {
	ID        string                     `json:"id"`
	OrderID   string                     `json:"order_id"`
	Items     []InventoryReservationItem `json:"items"`
	Status    ReservationStatus          `json:"status"`
	CreatedAt time.Time                  `json:"created_at"`
}

// InventoryReservationItem represents a single item in a reservation.
type InventoryReservationItem struct {
	ProductID string `json:"product_id"`
	Quantity  int    `json:"quantity"`
}

// ReservationStatus represents the state of an inventory reservation.
type ReservationStatus string

const (
	ReservationStatusPending   ReservationStatus = "PENDING"
	ReservationStatusConfirmed ReservationStatus = "CONFIRMED"
	ReservationStatusReleased  ReservationStatus = "RELEASED"
	ReservationStatusFailed    ReservationStatus = "FAILED"
)

// Shipment represents a shipping request.
type Shipment struct {
	ID              string         `json:"id"`
	OrderID         string         `json:"order_id"`
	CustomerID      string         `json:"customer_id"`
	ShippingAddress string         `json:"shipping_address"`
	Status          ShipmentStatus `json:"status"`
	TrackingNumber  string         `json:"tracking_number,omitempty"`
	CreatedAt       time.Time      `json:"created_at"`
}

// ShipmentStatus represents the state of a shipment.
type ShipmentStatus string

const (
	ShipmentStatusPending   ShipmentStatus = "PENDING"
	ShipmentStatusScheduled ShipmentStatus = "SCHEDULED"
	ShipmentStatusCancelled ShipmentStatus = "CANCELLED"
)

// Product represents an item in the inventory.
type Product struct {
	ID       string  `json:"id"`
	Name     string  `json:"name"`
	Price    float64 `json:"price"`
	Quantity int     `json:"quantity"` // Available quantity
}

// Customer represents a customer with payment information.
type Customer struct {
	ID              string  `json:"id"`
	Name            string  `json:"name"`
	Email           string  `json:"email"`
	Balance         float64 `json:"balance"` // Available balance for payments
	ShippingAddress string  `json:"shipping_address"`
}
