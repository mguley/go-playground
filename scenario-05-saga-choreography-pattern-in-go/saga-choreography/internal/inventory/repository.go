package inventory

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"

	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// Repository provides storage for products and reservations.
type Repository struct {
	mu           sync.RWMutex
	products     map[string]*models.Product
	reservations map[string]*models.InventoryReservation
}

// NewRepository creates a new inventory repository.
func NewRepository() *Repository {
	repo := &Repository{
		products:     make(map[string]*models.Product),
		reservations: make(map[string]*models.InventoryReservation),
	}

	repo.seedProducts()
	return repo
}

// seedProducts adds test product data.
func (r *Repository) seedProducts() {
	r.products["PROD-001"] = &models.Product{
		ID:       "PROD-001",
		Name:     "Laptop",
		Price:    999.99,
		Quantity: 10,
	}

	r.products["PROD-002"] = &models.Product{
		ID:       "PROD-002",
		Name:     "Wireless Mouse",
		Price:    29.99,
		Quantity: 100,
	}

	r.products["PROD-003"] = &models.Product{
		ID:       "PROD-003",
		Name:     "USB-C Cable",
		Price:    19.99,
		Quantity: 2, // Low stock to test failure scenarios
	}

	r.products["PROD-004"] = &models.Product{
		ID:       "PROD-004",
		Name:     "Mechanical Keyboard",
		Price:    149.99,
		Quantity: 0, // Out of stock to demonstrate inventory failure
	}
}

// GetProduct retrieves a product by ID.
func (r *Repository) GetProduct(id string) (*models.Product, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	product, exists := r.products[id]
	if !exists {
		return nil, fmt.Errorf("product %s not found", id)
	}

	return new(*product), nil
}

// ReserveInventory attempts to reserve items for an order.
// This operation is atomic: either all items are reserved or none are.
// This prevents partial reservations that would be difficult to compensate.
func (r *Repository) ReserveInventory(ctx context.Context, orderID string, items []models.InventoryReservationItem) (*models.InventoryReservation, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// First pass: validate that all items are available.
	// We check everything before modifying anything to maintain atomicity.
	for _, item := range items {
		product, exists := r.products[item.ProductID]
		if !exists {
			return nil, fmt.Errorf("product %s not found", item.ProductID)
		}
		if product.Quantity < item.Quantity {
			return nil, fmt.Errorf("insufficient stock for %s: available %d, requested %d",
				product.Name, product.Quantity, item.Quantity)
		}
	}

	// Second pass: all items available, so deduct quantities and create the reservation.
	reservation := &models.InventoryReservation{
		ID:        events.GenerateID("RES"),
		OrderID:   orderID,
		Items:     items,
		Status:    models.ReservationStatusConfirmed,
		CreatedAt: time.Now(),
	}

	for _, item := range items {
		r.products[item.ProductID].Quantity -= item.Quantity
	}

	r.reservations[reservation.ID] = reservation
	return reservation, nil
}

// GetReservationByOrderID retrieves a reservation by order ID.
func (r *Repository) GetReservationByOrderID(orderID string) (*models.InventoryReservation, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, reservation := range r.reservations {
		if reservation.OrderID == orderID {
			return new(*reservation), nil
		}
	}

	return nil, fmt.Errorf("reservation for order %s not found", orderID)
}

// ReleaseReservation releases a reservation and returns items to stock.
// This is the compensating transaction for inventory reservation.
// Note that this method is idempotent: releasing an already-released reservation
// is a safe no-op. This is important because compensating transactions may be
// triggered more than once in the face of retries or duplicate events.
func (r *Repository) ReleaseReservation(ctx context.Context, id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	reservation, exists := r.reservations[id]
	if !exists {
		return fmt.Errorf("reservation %s not found", id)
	}

	// Idempotent: releasing an already-released reservation is a no-op.
	if reservation.Status == models.ReservationStatusReleased {
		return nil
	}

	// Return items to stock
	for _, item := range reservation.Items {
		if product, exists := r.products[item.ProductID]; exists {
			product.Quantity += item.Quantity
		}
	}

	reservation.Status = models.ReservationStatusReleased
	return nil
}

// AllReservations returns an iterator over all reservations.
func (r *Repository) AllReservations() iter.Seq[*models.InventoryReservation] {
	return func(yield func(*models.InventoryReservation) bool) {
		r.mu.RLock()
		defer r.mu.RUnlock()

		for _, res := range r.reservations {
			if !yield(new(*res)) {
				return
			}
		}
	}
}
