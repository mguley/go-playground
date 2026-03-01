package inventory

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// Service handles inventory management.
//
// This service demonstrates an important choreography pattern: maintaining a local
// cache of data from upstream events. The Inventory service needs order item details
// to reserve stock, but those details originate in the Order service. Rather than
// querying the Order service directly (which would introduce coupling and defeat
// the purpose of choreography), the Inventory service subscribes to "order.created"
// events and caches the item details locally. When "payment.completed" arrives later,
// the cached items are already available.
type Service struct {
	repo     *Repository
	eventBus *events.EventBus
	logger   *slog.Logger

	// orderItemsCache stores order items keyed by order ID.
	// This is populated when "order.created" events arrive and consumed when
	// "payment.completed" events trigger inventory reservation.
	// In production, this would be a local database table or an event-sourced view.
	orderItemsCache sync.Map

	processedEvents sync.Map // See the memory leak warning in order/service.go
}

// NewService creates a new Inventory service.
func NewService(repo *Repository, eventBus *events.EventBus, logger *slog.Logger) *Service {
	s := &Service{
		repo:     repo,
		eventBus: eventBus,
		logger:   logger,
	}

	s.subscribeToEvents()
	return s
}

// subscribeToEvents sets up event handlers.
func (s *Service) subscribeToEvents() {
	// The Inventory service subscribes to three event types:
	// 1. order.created: to cache order items for later use
	// 2. payment.completed: to actually reserve inventory (only after payment succeeds)
	// 3. order.cancelled: to release any reservations (compensating transaction)
	eventTypes := []events.EventType{
		events.OrderCreated,
		events.PaymentCompleted,
		events.OrderCancelled,
	}

	s.eventBus.Subscribe("inventory-service", eventTypes, s.handleEvent)
}

// isProcessed provides idempotency.
func (s *Service) isProcessed(eventID string) bool {
	_, loaded := s.processedEvents.LoadOrStore(eventID, true)
	return loaded
}

// handleEvent processes incoming events.
func (s *Service) handleEvent(ctx context.Context, event *events.Event) error {
	if s.isProcessed(event.ID) {
		s.logger.Info("[InventoryService] Skipping duplicate event",
			"service", "inventory",
			"event_id", event.ID,
		)
		return nil
	}

	s.logger.Info("[InventoryService] Received event",
		"service", "inventory",
		"event_type", string(event.Type),
		"correlation_id", event.CorrelationID,
	)

	switch event.Type {
	case events.OrderCreated:
		return s.handleOrderCreated(ctx, event)
	case events.PaymentCompleted:
		return s.handlePaymentCompleted(ctx, event)
	case events.OrderCancelled:
		return s.handleOrderCancelled(ctx, event)
	default:
		return nil
	}
}

// handleOrderCreated caches order items for later reservation.
// The Inventory service doesn't act on the order yet (it waits for payment confirmation),
// but it needs to remember what items were ordered so it can reserve them later.
func (s *Service) handleOrderCreated(ctx context.Context, event *events.Event) error {
	var data events.OrderCreatedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse order created data: %w", err)
	}

	// Convert order items to inventory reservation items and cache them
	items := make([]models.InventoryReservationItem, len(data.Order.Items))
	for i, orderItem := range data.Order.Items {
		items[i] = models.InventoryReservationItem{
			ProductID: orderItem.ProductID,
			Quantity:  orderItem.Quantity,
		}
	}

	s.orderItemsCache.Store(data.Order.ID, items)

	s.logger.Info("[InventoryService] Cached order items (awaiting payment)",
		"service", "inventory",
		"order_id", data.Order.ID,
		"item_count", len(items),
	)

	return nil
}

// handlePaymentCompleted reserves inventory after payment is confirmed.
func (s *Service) handlePaymentCompleted(ctx context.Context, event *events.Event) error {
	var data events.PaymentCompletedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse payment completed data: %w", err)
	}

	s.logger.Info("[InventoryService] Reserving inventory",
		"service", "inventory",
		"order_id", data.OrderID,
	)

	// Retrieve cached order items
	cachedItems, ok := s.orderItemsCache.Load(data.OrderID)
	if !ok {
		// This shouldn't happen in normal flow, but we handle it defensively
		s.logger.Error("[InventoryService] No cached items found for order",
			"service", "inventory",
			"order_id", data.OrderID,
		)

		failedEvent, _ := events.NewEvent(
			events.InventoryReserveFailed,
			data.OrderID,
			"inventory",
			events.InventoryReserveFailedData{
				OrderID: data.OrderID,
				Reason:  "order items not found in local cache",
			},
		)
		failedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
		return s.eventBus.Publish(ctx, failedEvent)
	}

	items := cachedItems.([]models.InventoryReservationItem)

	reservation, err := s.repo.ReserveInventory(ctx, data.OrderID, items)
	if err != nil {
		// Reservation failed: publish failure event
		s.logger.Warn("[InventoryService] Reservation failed",
			"service", "inventory",
			"order_id", data.OrderID,
			"error", err,
		)

		failedEvent, err := events.NewEvent(
			events.InventoryReserveFailed,
			data.OrderID,
			"inventory",
			events.InventoryReserveFailedData{
				OrderID: data.OrderID,
				Reason:  err.Error(),
			},
		)
		if err != nil {
			return fmt.Errorf("failed to create inventory reserve failed event: %w", err)
		}

		failedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
		return s.eventBus.Publish(ctx, failedEvent)
	}

	// Clean up cache after successful reservation
	s.orderItemsCache.Delete(data.OrderID)

	s.logger.Info("[InventoryService] Inventory reserved",
		"service", "inventory",
		"order_id", data.OrderID,
		"reservation_id", reservation.ID,
	)

	// Publish reservation success event
	reservedEvent, err := events.NewEvent(
		events.InventoryReserved,
		data.OrderID,
		"inventory",
		events.InventoryReservedData{
			ReservationID: reservation.ID,
			OrderID:       data.OrderID,
			Items:         reservation.Items,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create inventory reserved event: %w", err)
	}

	reservedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
	return s.eventBus.Publish(ctx, reservedEvent)
}

// handleOrderCancelled releases the reservation as a compensating transaction.
func (s *Service) handleOrderCancelled(ctx context.Context, event *events.Event) error {
	var data events.OrderCancelledData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse order cancelled data: %w", err)
	}

	// Clean up any cached items for this order
	s.orderItemsCache.Delete(data.OrderID)

	// Find the reservation for this order
	reservation, err := s.repo.GetReservationByOrderID(data.OrderID)
	if err != nil {
		// No reservation found: nothing to release.
		// Normal when the saga failed before inventory was reserved.
		s.logger.Info("[InventoryService] No reservation found for cancelled order",
			"service", "inventory",
			"order_id", data.OrderID,
		)
		return nil
	}

	// Only release confirmed reservations
	if reservation.Status != models.ReservationStatusConfirmed {
		s.logger.Info("[InventoryService] Reservation not confirmed, skipping release",
			"service", "inventory",
			"reservation_id", reservation.ID,
			"status", string(reservation.Status),
		)
		return nil
	}

	// Release the reservation (compensating transaction)
	s.logger.Info("[InventoryService] Releasing reservation",
		"service", "inventory",
		"reservation_id", reservation.ID,
		"order_id", data.OrderID,
	)

	if err = s.repo.ReleaseReservation(ctx, reservation.ID); err != nil {
		return fmt.Errorf("failed to release reservation: %w", err)
	}

	s.logger.Info("[InventoryService] Reservation released",
		"service", "inventory",
		"order_id", data.OrderID,
	)

	// Publish release event
	releasedEvent, err := events.NewEvent(
		events.InventoryReleased,
		data.OrderID,
		"inventory",
		events.InventoryReleasedData{
			ReservationID: reservation.ID,
			OrderID:       data.OrderID,
			Reason:        data.Reason,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create inventory released event: %w", err)
	}

	releasedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
	return s.eventBus.Publish(ctx, releasedEvent)
}

// GetProduct retrieves product information.
func (s *Service) GetProduct(id string) (*models.Product, error) {
	return s.repo.GetProduct(id)
}
