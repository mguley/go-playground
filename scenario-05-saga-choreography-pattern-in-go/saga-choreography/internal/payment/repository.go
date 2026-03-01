package payment

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"

	"saga-choreography/pkg/models"
)

// Repository provides storage for payments and customers.
type Repository struct {
	mu        sync.RWMutex
	payments  map[string]*models.Payment
	customers map[string]*models.Customer
}

// NewRepository creates a new payment repository.
func NewRepository() *Repository {
	repo := &Repository{
		payments:  make(map[string]*models.Payment),
		customers: make(map[string]*models.Customer),
	}

	// Seed with test customers that exercise different scenarios
	repo.seedCustomers()

	return repo
}

// seedCustomers adds test customer data.
func (r *Repository) seedCustomers() {
	r.customers["CUST-001"] = &models.Customer{
		ID:              "CUST-001",
		Name:            "John Doe",
		Email:           "john@example.com",
		Balance:         1000.00,
		ShippingAddress: "123 Main St, New York, NY 10001",
	}

	r.customers["CUST-002"] = &models.Customer{
		ID:              "CUST-002",
		Name:            "Jane Smith",
		Email:           "jane@example.com",
		Balance:         50.00, // Low balance to test payment failure scenarios
		ShippingAddress: "456 Oak Ave, Los Angeles, CA 90001",
	}

	r.customers["CUST-003"] = &models.Customer{
		ID:              "CUST-003",
		Name:            "Bob Wilson",
		Email:           "bob@example.com",
		Balance:         5000.00,
		ShippingAddress: "789 Pine Rd, Chicago, IL 60601",
	}
}

// CreatePayment stores a new payment.
func (r *Repository) CreatePayment(ctx context.Context, payment *models.Payment) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	payment.CreatedAt = time.Now()
	r.payments[payment.ID] = payment
	return nil
}

// GetPayment retrieves a payment by ID.
func (r *Repository) GetPayment(ctx context.Context, id string) (*models.Payment, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	payment, exists := r.payments[id]
	if !exists {
		return nil, fmt.Errorf("payment %s not found", id)
	}

	return new(*payment), nil
}

// GetPaymentByOrderID retrieves a payment by order ID.
func (r *Repository) GetPaymentByOrderID(orderID string) (*models.Payment, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, payment := range r.payments {
		if payment.OrderID == orderID {
			return new(*payment), nil
		}
	}

	return nil, fmt.Errorf("payment for order %s not found", orderID)
}

// UpdatePaymentStatus updates a payment's status.
func (r *Repository) UpdatePaymentStatus(ctx context.Context, id string, status models.PaymentStatus) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	payment, exists := r.payments[id]
	if !exists {
		return fmt.Errorf("payment %s not found", id)
	}

	payment.Status = status
	return nil
}

// GetCustomer retrieves a customer by ID.
func (r *Repository) GetCustomer(id string) (*models.Customer, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	customer, exists := r.customers[id]
	if !exists {
		return nil, fmt.Errorf("customer %s not found", id)
	}

	return new(*customer), nil
}

// DeductBalance deducts an amount from a customer's balance.
func (r *Repository) DeductBalance(ctx context.Context, customerID string, amount float64) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	customer, exists := r.customers[customerID]
	if !exists {
		return fmt.Errorf("customer %s not found", customerID)
	}

	if customer.Balance < amount {
		return fmt.Errorf("insufficient balance: available $%.2f, required $%.2f",
			customer.Balance, amount)
	}

	customer.Balance -= amount
	return nil
}

// CreditBalance adds an amount to a customer's balance (for refunds).
// NOTE: This method is not idempotent - calling it twice doubles the credit.
// Idempotency is enforced at the service layer via processedEvents. In production,
// you'd also want a database-level idempotency check (e.g., a unique refund
// transaction ID) to guard against duplicate refunds after service restarts.
func (r *Repository) CreditBalance(ctx context.Context, customerID string, amount float64) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	customer, exists := r.customers[customerID]
	if !exists {
		return fmt.Errorf("customer %s not found", customerID)
	}

	customer.Balance += amount
	return nil
}

// AllPayments returns an iterator over all payments.
func (r *Repository) AllPayments() iter.Seq[*models.Payment] {
	return func(yield func(*models.Payment) bool) {
		r.mu.RLock()
		defer r.mu.RUnlock()

		for _, payment := range r.payments {
			if !yield(new(*payment)) {
				return
			}
		}
	}
}
