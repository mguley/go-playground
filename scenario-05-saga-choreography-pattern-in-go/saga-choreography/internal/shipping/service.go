package shipping

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// Service handles shipping operations.
type Service struct {
	repo            *Repository
	eventBus        *events.EventBus
	logger          *slog.Logger
	simulateFailure bool     // For testing failure scenarios
	processedEvents sync.Map // See the memory leak warning in order/service.go

	// orderCustomerCache stores customer and address data from order events.
	// Same pattern as the Inventory service: cache data from upstream events
	// rather than querying the Order service directly.
	orderCustomerCache sync.Map
}

// customerInfo holds the subset of order data needed for shipping.
type customerInfo struct {
	CustomerID      string
	ShippingAddress string
}

// NewService creates a new Shipping service.
func NewService(repo *Repository, eventBus *events.EventBus, logger *slog.Logger) *Service {
	s := &Service{
		repo:     repo,
		eventBus: eventBus,
		logger:   logger,
	}

	s.subscribeToEvents()
	return s
}

// SetSimulateFailure enables/disables failure simulation for testing.
func (s *Service) SetSimulateFailure(simulate bool) {
	s.simulateFailure = simulate
}

// subscribeToEvents sets up event handlers.
func (s *Service) subscribeToEvents() {
	// The Shipping service subscribes to:
	// 1. order.created: to cache customer and address data
	// 2. inventory.reserved: to schedule shipping (only after stock is confirmed)
	// 3. order.cancelled: to cancel any scheduled shipments
	eventTypes := []events.EventType{
		events.OrderCreated,
		events.InventoryReserved,
		events.OrderCancelled,
	}

	s.eventBus.Subscribe("shipping-service", eventTypes, s.handleEvent)
}

// isProcessed provides idempotency.
func (s *Service) isProcessed(eventID string) bool {
	_, loaded := s.processedEvents.LoadOrStore(eventID, true)
	return loaded
}

// handleEvent processes incoming events.
func (s *Service) handleEvent(ctx context.Context, event *events.Event) error {
	if s.isProcessed(event.ID) {
		s.logger.Info("[ShippingService] Skipping duplicate event",
			"service", "shipping",
			"event_id", event.ID,
		)
		return nil
	}

	s.logger.Info("[ShippingService] Received event",
		"service", "shipping",
		"event_type", string(event.Type),
		"correlation_id", event.CorrelationID,
	)

	switch event.Type {
	case events.OrderCreated:
		return s.handleOrderCreated(ctx, event)
	case events.InventoryReserved:
		return s.handleInventoryReserved(ctx, event)
	case events.OrderCancelled:
		return s.handleOrderCancelled(ctx, event)
	default:
		return nil
	}
}

// handleOrderCreated caches customer and address data for later use.
func (s *Service) handleOrderCreated(ctx context.Context, event *events.Event) error {
	var data events.OrderCreatedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse order created data: %w", err)
	}

	// Cache customer info so it's available when we schedule shipping.
	// In a production system, the order event would carry the shipping address
	// selected during checkout. For this demo, we use a default address.
	s.orderCustomerCache.Store(data.Order.ID, customerInfo{
		CustomerID:      data.Order.CustomerID,
		ShippingAddress: "Default Address",
	})

	s.logger.Info("[ShippingService] Cached customer info",
		"service", "shipping",
		"order_id", data.Order.ID,
	)
	return nil
}

// handleInventoryReserved schedules shipping after inventory is reserved.
func (s *Service) handleInventoryReserved(ctx context.Context, event *events.Event) error {
	var data events.InventoryReservedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse inventory reserved data: %w", err)
	}

	s.logger.Info("[ShippingService] Scheduling shipment",
		"service", "shipping",
		"order_id", data.OrderID,
	)

	// Simulate occasional failures for testing
	if s.simulateFailure {
		s.logger.Warn("[ShippingService] Simulating shipping failure",
			"service", "shipping",
			"order_id", data.OrderID,
		)

		failedEvent, err := events.NewEvent(
			events.ShipmentScheduleFailed,
			data.OrderID,
			"shipping",
			events.ShipmentScheduleFailedData{
				OrderID: data.OrderID,
				Reason:  "Shipping carrier unavailable",
			},
		)
		if err != nil {
			return fmt.Errorf("failed to create shipment failed event: %w", err)
		}

		failedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
		return s.eventBus.Publish(ctx, failedEvent)
	}

	// Retrieve cached customer info
	customerID := data.OrderID // fallback
	address := "Default Address"
	if cached, ok := s.orderCustomerCache.Load(data.OrderID); ok {
		info := cached.(customerInfo)
		customerID = info.CustomerID
		address = info.ShippingAddress
	}

	// Create shipment
	shipment := &models.Shipment{
		ID:              events.GenerateID("SHIP"),
		OrderID:         data.OrderID,
		CustomerID:      customerID,
		ShippingAddress: address,
		Status:          models.ShipmentStatusScheduled,
		TrackingNumber:  generateTrackingNumber(),
	}

	if err := s.repo.CreateShipment(ctx, shipment); err != nil {
		return fmt.Errorf("failed to create shipment: %w", err)
	}

	// Clean up cache
	s.orderCustomerCache.Delete(data.OrderID)

	s.logger.Info("[ShippingService] Shipment scheduled",
		"service", "shipping",
		"order_id", data.OrderID,
		"shipment_id", shipment.ID,
		"tracking_number", shipment.TrackingNumber,
	)

	// Publish shipment scheduled event: this completes the saga!
	scheduledEvent, err := events.NewEvent(
		events.ShipmentScheduled,
		data.OrderID,
		"shipping",
		events.ShipmentScheduledData{
			ShipmentID:     shipment.ID,
			OrderID:        data.OrderID,
			TrackingNumber: shipment.TrackingNumber,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create shipment scheduled event: %w", err)
	}

	scheduledEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
	return s.eventBus.Publish(ctx, scheduledEvent)
}

// handleOrderCancelled cancels the shipment as a compensating transaction.
func (s *Service) handleOrderCancelled(ctx context.Context, event *events.Event) error {
	var data events.OrderCancelledData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse order cancelled data: %w", err)
	}

	// Clean up any cached data
	s.orderCustomerCache.Delete(data.OrderID)

	// Find the shipment for this order
	shipment, err := s.repo.GetShipmentByOrderID(data.OrderID)
	if err != nil {
		// No shipment found: nothing to cancel.
		// Normal when the saga failed before shipping was scheduled.
		s.logger.Info("[ShippingService] No shipment found for cancelled order",
			"service", "shipping",
			"order_id", data.OrderID,
		)
		return nil
	}

	// Only cancel scheduled shipments
	if shipment.Status != models.ShipmentStatusScheduled {
		s.logger.Info("[ShippingService] Shipment not scheduled, skipping cancellation",
			"service", "shipping",
			"shipment_id", shipment.ID,
			"status", string(shipment.Status),
		)
		return nil
	}

	// Cancel the shipment (compensating transaction)
	s.logger.Info("[ShippingService] Cancelling shipment",
		"service", "shipping",
		"shipment_id", shipment.ID,
		"order_id", data.OrderID,
	)

	if err = s.repo.UpdateShipmentStatus(ctx, shipment.ID, models.ShipmentStatusCancelled); err != nil {
		return fmt.Errorf("failed to cancel shipment: %w", err)
	}

	s.logger.Info("[ShippingService] Shipment cancelled",
		"service", "shipping",
		"order_id", data.OrderID,
	)

	// Publish cancellation event
	cancelledEvent, err := events.NewEvent(
		events.ShipmentCancelled,
		data.OrderID,
		"shipping",
		events.ShipmentCancelledData{
			ShipmentID: shipment.ID,
			OrderID:    data.OrderID,
			Reason:     data.Reason,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create shipment cancelled event: %w", err)
	}

	cancelledEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
	return s.eventBus.Publish(ctx, cancelledEvent)
}
