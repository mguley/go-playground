package shipping

import (
	"context"
	"fmt"
	"sync"
	"time"

	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// Repository provides storage for shipments.
type Repository struct {
	mu        sync.RWMutex
	shipments map[string]*models.Shipment
}

// NewRepository creates a new shipping repository.
func NewRepository() *Repository {
	return &Repository{
		shipments: make(map[string]*models.Shipment),
	}
}

// CreateShipment stores a new shipment.
func (r *Repository) CreateShipment(ctx context.Context, shipment *models.Shipment) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	shipment.CreatedAt = time.Now()
	r.shipments[shipment.ID] = shipment
	return nil
}

// GetShipment retrieves a shipment by ID.
func (r *Repository) GetShipment(ctx context.Context, id string) (*models.Shipment, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	shipment, exists := r.shipments[id]
	if !exists {
		return nil, fmt.Errorf("shipment %s not found", id)
	}

	return new(*shipment), nil
}

// GetShipmentByOrderID retrieves a shipment by order ID.
func (r *Repository) GetShipmentByOrderID(orderID string) (*models.Shipment, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, shipment := range r.shipments {
		if shipment.OrderID == orderID {
			return new(*shipment), nil
		}
	}

	return nil, fmt.Errorf("shipment for order %s not found", orderID)
}

// UpdateShipmentStatus updates a shipment's status.
func (r *Repository) UpdateShipmentStatus(ctx context.Context, id string, status models.ShipmentStatus) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	shipment, exists := r.shipments[id]
	if !exists {
		return fmt.Errorf("shipment %s not found", id)
	}

	shipment.Status = status
	return nil
}

// generateTrackingNumber creates a random tracking number.
func generateTrackingNumber() string {
	return events.GenerateID("TRACK")
}
