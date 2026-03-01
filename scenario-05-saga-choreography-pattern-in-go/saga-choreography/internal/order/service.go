package order

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"sync"

	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// Service handles order related business logic and saga coordination.
// In a choreography, the Order service plays a dual role: it initiates the saga
// by creating orders and publishing the first event, and it tracks saga progress
// by listening to success/failure events from all other services.
type Service struct {
	repo     *Repository
	eventBus *events.EventBus
	logger   *slog.Logger

	// processedEvents provides idempotency protection. In a distributed system,
	// the same event might be delivered more than once (at-least-once delivery).
	// Without this guard, a duplicate "payment.completed" event could corrupt
	// the order state or trigger duplicate downstream actions.
	//
	// WARNING: In a long-running production service, this map grows without bound
	// and will eventually cause an OOM crash. For production use, replace this with:
	//   - An LRU cache (e.g., hashicorp/golang-lru) with a size limit
	//   - A Redis key with a TTL (e.g., SET event_id 1 EX 3600 NX)
	//   - The Transactional Inbox Pattern: store processed event IDs in the
	//     service's primary database with a unique constraint, gaining both
	//     idempotency and atomicity with the business operation
	processedEvents sync.Map
}

// NewService creates a new Order service.
func NewService(repo *Repository, eventBus *events.EventBus, logger *slog.Logger) *Service {
	s := &Service{
		repo:     repo,
		eventBus: eventBus,
		logger:   logger,
	}

	// Subscribe to events that affect order status
	s.subscribeToEvents()

	return s
}

// subscribeToEvents sets up event handlers for the Order service.
func (s *Service) subscribeToEvents() {
	// The Order service listens to events from other services to track saga progress.
	// Notice that it subscribes to both success and failure events from every service,
	// because it needs to know the outcome of each step.
	eventTypes := []events.EventType{
		events.PaymentCompleted,
		events.PaymentFailed,
		events.InventoryReserved,
		events.InventoryReserveFailed,
		events.ShipmentScheduled,
		events.ShipmentScheduleFailed,
		// Compensation events are tracked for observability
		events.PaymentRefunded,
		events.InventoryReleased,
	}

	s.eventBus.Subscribe("order-service", eventTypes, s.handleEvent)
}

// isProcessed checks if an event has already been processed and marks it if not.
// This provides exactly once processing semantics on top of at-least-once delivery.
func (s *Service) isProcessed(eventID string) bool {
	_, loaded := s.processedEvents.LoadOrStore(eventID, true)
	return loaded
}

// handleEvent processes incoming events.
func (s *Service) handleEvent(ctx context.Context, event *events.Event) error {
	// Idempotency check: skip events we've already processed
	if s.isProcessed(event.ID) {
		s.logger.Info("[OrderService] Skipping duplicate event",
			"service", "order",
			"event_id", event.ID,
		)
		return nil
	}

	s.logger.Info("[OrderService] Received event",
		"service", "order",
		"event_type", string(event.Type),
		"correlation_id", event.CorrelationID,
	)

	switch event.Type {
	case events.PaymentCompleted:
		return s.handlePaymentCompleted(ctx, event)
	case events.PaymentFailed:
		return s.handlePaymentFailed(ctx, event)
	case events.InventoryReserved:
		return s.handleInventoryReserved(ctx, event)
	case events.InventoryReserveFailed:
		return s.handleInventoryReserveFailed(ctx, event)
	case events.ShipmentScheduled:
		return s.handleShipmentScheduled(ctx, event)
	case events.ShipmentScheduleFailed:
		return s.handleShipmentScheduleFailed(ctx, event)
	case events.PaymentRefunded:
		return s.handlePaymentRefunded(ctx, event)
	case events.InventoryReleased:
		return s.handleInventoryReleased(ctx, event)
	default:
		s.logger.Warn("[OrderService] Unknown event type",
			"service", "order",
			"event_type", string(event.Type),
		)
		return nil
	}
}

// CreateOrder initiates the order saga.
func (s *Service) CreateOrder(ctx context.Context, customerID string, items []models.OrderItem) (*models.Order, error) {
	// Calculate total amount
	var totalAmount float64
	for _, item := range items {
		totalAmount += item.Price * float64(item.Quantity)
	}

	// Create the order
	order := &models.Order{
		ID:          generateOrderID(),
		CustomerID:  customerID,
		Items:       items,
		TotalAmount: totalAmount,
		Status:      models.OrderStatusPending,
	}

	if err := s.repo.Create(ctx, order); err != nil {
		return nil, fmt.Errorf("failed to create order: %w", err)
	}

	s.logger.Info("[OrderService] Created order",
		"service", "order",
		"order_id", order.ID,
		"customer_id", customerID,
		"total_amount", totalAmount,
	)

	// Publish the OrderCreated event to start the saga.
	// The order ID serves as the correlation ID for the entire saga, linking
	// every event across all services back to this originating order.
	event, err := events.NewEvent(
		events.OrderCreated,
		order.ID,
		"order",
		events.OrderCreatedData{Order: *order},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create event: %w", err)
	}

	event.WithCorrelation(order.ID)

	if err = s.eventBus.Publish(ctx, event); err != nil {
		return nil, fmt.Errorf("failed to publish event: %w", err)
	}

	return order, nil
}

// Each handler follows the same pattern:
//   1. Parse the event data.
//   2. Call RecordStepResult with the step name, result, and any resource refs.
//   3. Log the outcome.
//   4. Call actOnSaga to publish the appropriate terminal event if the saga
//      has reached a terminal state.
//
// Notice that no handler checks or cares about the current order status.
// The repository's RecordStepResult method handles all concurrency concerns
// atomically. If the order is already in a terminal state (because a racing
// event got there first), RecordStepResult returns SagaActionNone and the
// handler simply does nothing.

// handlePaymentCompleted records that payment succeeded.
func (s *Service) handlePaymentCompleted(ctx context.Context, event *events.Event) error {
	var data events.PaymentCompletedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse payment completed data: %w", err)
	}

	action, err := s.repo.RecordStepResult(ctx, data.OrderID,
		StepPayment, models.StepSucceeded,
		StepRefs{PaymentID: data.PaymentID},
	)
	if err != nil {
		return err
	}

	s.logger.Info("[OrderService] Payment completed",
		"service", "order",
		"order_id", data.OrderID,
		"payment_id", data.PaymentID,
	)

	return s.actOnSaga(ctx, event, data.OrderID, action, "")
}

// handlePaymentFailed handles payment failure. The saga should be rolled back.
func (s *Service) handlePaymentFailed(ctx context.Context, event *events.Event) error {
	var data events.PaymentFailedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse payment failed data: %w", err)
	}

	action, err := s.repo.RecordStepResult(ctx, data.OrderID,
		StepPayment, models.StepFailed,
		StepRefs{},
	)
	if err != nil {
		return err
	}

	s.logger.Warn("[OrderService] Payment failed",
		"service", "order",
		"order_id", data.OrderID,
		"reason", data.Reason,
	)

	return s.actOnSaga(ctx, event, data.OrderID, action, "Payment failed: "+data.Reason)
}

// handleInventoryReserved records that inventory was reserved.
func (s *Service) handleInventoryReserved(ctx context.Context, event *events.Event) error {
	var data events.InventoryReservedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse inventory reserved data: %w", err)
	}

	action, err := s.repo.RecordStepResult(ctx, data.OrderID,
		StepInventory, models.StepSucceeded,
		StepRefs{ReservationID: data.ReservationID},
	)
	if err != nil {
		return err
	}

	s.logger.Info("[OrderService] Inventory reserved",
		"service", "order",
		"order_id", data.OrderID,
		"reservation_id", data.ReservationID,
	)

	return s.actOnSaga(ctx, event, data.OrderID, action, "")
}

// handleInventoryReserveFailed records that inventory reservation failed.
func (s *Service) handleInventoryReserveFailed(ctx context.Context, event *events.Event) error {
	var data events.InventoryReserveFailedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse inventory reserve failed data: %w", err)
	}

	action, err := s.repo.RecordStepResult(ctx, data.OrderID,
		StepInventory, models.StepFailed,
		StepRefs{},
	)
	if err != nil {
		return err
	}

	s.logger.Warn("[OrderService] Inventory reservation failed",
		"service", "order",
		"order_id", data.OrderID,
		"reason", data.Reason,
	)

	return s.actOnSaga(ctx, event, data.OrderID, action, "Inventory reservation failed: "+data.Reason)
}

// handleShipmentScheduled records that shipping was scheduled.
func (s *Service) handleShipmentScheduled(ctx context.Context, event *events.Event) error {
	var data events.ShipmentScheduledData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse shipment scheduled data: %w", err)
	}

	action, err := s.repo.RecordStepResult(ctx, data.OrderID,
		StepShipping, models.StepSucceeded,
		StepRefs{
			ShipmentID:     data.ShipmentID,
			TrackingNumber: data.TrackingNumber,
		},
	)
	if err != nil {
		return err
	}

	s.logger.Info("[OrderService] Shipment scheduled",
		"service", "order",
		"order_id", data.OrderID,
		"shipment_id", data.ShipmentID,
		"tracking_number", data.TrackingNumber,
	)

	return s.actOnSaga(ctx, event, data.OrderID, action, "")
}

// handleShipmentScheduleFailed records that shipping scheduling failed.
func (s *Service) handleShipmentScheduleFailed(ctx context.Context, event *events.Event) error {
	var data events.ShipmentScheduleFailedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse shipment schedule failed data: %w", err)
	}

	action, err := s.repo.RecordStepResult(ctx, data.OrderID,
		StepShipping, models.StepFailed,
		StepRefs{},
	)
	if err != nil {
		return err
	}

	s.logger.Warn("[OrderService] Shipment scheduling failed",
		"service", "order",
		"order_id", data.OrderID,
		"reason", data.Reason,
	)

	return s.actOnSaga(ctx, event, data.OrderID, action, "Shipping failed: "+data.Reason)
}

// handlePaymentRefunded logs the refund completion.
func (s *Service) handlePaymentRefunded(ctx context.Context, event *events.Event) error {
	var data events.PaymentRefundedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse payment refunded data: %w", err)
	}

	s.repo.MarkStepCompensated(ctx, data.OrderID, StepPayment)

	s.logger.Info("[OrderService] Payment refunded",
		"service", "order",
		"order_id", data.OrderID,
		"amount", data.Amount,
	)

	return nil
}

// handleInventoryReleased logs the inventory release.
func (s *Service) handleInventoryReleased(ctx context.Context, event *events.Event) error {
	var data events.InventoryReleasedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse inventory released data: %w", err)
	}

	s.repo.MarkStepCompensated(ctx, data.OrderID, StepInventory)

	s.logger.Info("[OrderService] Inventory released",
		"service", "order",
		"order_id", data.OrderID,
	)

	return nil
}

// actOnSaga publishes the appropriate terminal event based on the action
// returned by RecordStepResult.
//
// This method is the single point where terminal events are published.
// Because RecordStepResult returns a non-None action at most once per saga
// (the terminal check in the repository prevents subsequent calls from
// returning an action), exactly one terminal event is published per saga,
// even under concurrent event delivery.
func (s *Service) actOnSaga(ctx context.Context, causingEvent *events.Event, orderID string, action models.SagaAction, reason string) error {
	switch action {
	case models.SagaActionComplete:
		return s.completeOrder(ctx, causingEvent, orderID)
	case models.SagaActionCancel:
		return s.cancelOrder(ctx, causingEvent, orderID, reason)
	default:
		return nil
	}
}

// completeOrder publishes an order.completed event.
func (s *Service) completeOrder(ctx context.Context, causingEvent *events.Event, orderID string) error {
	// Fetch the order to get all the resource references for the completion event.
	order, err := s.repo.Get(ctx, orderID)
	if err != nil {
		return fmt.Errorf("failed to get order for completion: %w", err)
	}

	s.logger.Info("[OrderService] SAGA COMPLETED",
		"service", "order",
		"order_id", orderID,
		"tracking_number", order.TrackingNumber,
	)

	completedEvent, err := events.NewEvent(
		events.OrderCompleted,
		orderID,
		"order",
		events.OrderCompletedData{
			OrderID:        orderID,
			PaymentID:      order.PaymentID,
			ReservationID:  order.ReservationID,
			ShipmentID:     order.ShipmentID,
			TrackingNumber: order.TrackingNumber,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create completed event: %w", err)
	}

	completedEvent.WithCorrelation(causingEvent.CorrelationID).WithCausation(causingEvent.ID)
	return s.eventBus.Publish(ctx, completedEvent)
}

// cancelOrder publishes an order.cancelled event.
// This is the trigger for compensating transactions across all services.
// Each service that receives this event checks its own state and undoes
// whatever work it completed for this order.
func (s *Service) cancelOrder(ctx context.Context, causingEvent *events.Event, orderID, reason string) error {
	s.logger.Warn("[OrderService] SAGA CANCELLED",
		"service", "order",
		"order_id", orderID,
		"reason", reason,
	)

	cancelledEvent, err := events.NewEvent(
		events.OrderCancelled,
		orderID,
		"order",
		events.OrderCancelledData{
			OrderID: orderID,
			Reason:  reason,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create cancelled event: %w", err)
	}

	cancelledEvent.WithCorrelation(causingEvent.CorrelationID).WithCausation(causingEvent.ID)
	return s.eventBus.Publish(ctx, cancelledEvent)
}

// GetOrder retrieves an order by ID.
func (s *Service) GetOrder(id string) (*models.Order, error) {
	return s.repo.Get(context.Background(), id)
}

// GetAllOrders returns an iterator over all orders.
func (s *Service) GetAllOrders() iter.Seq[*models.Order] {
	return s.repo.All()
}

// generateOrderID creates a unique order ID.
func generateOrderID() string {
	return events.GenerateID("ORD")
}
