package payment

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"saga-choreography/pkg/events"
	"saga-choreography/pkg/models"
)

// Service handles payment processing and refunds.
type Service struct {
	repo            *Repository
	eventBus        *events.EventBus
	logger          *slog.Logger
	processedEvents sync.Map // See the memory leak warning in order/service.go
}

// NewService creates a new Payment service.
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
	// Listen for order created to process payment.
	// Listen for order cancelled to issue refund (compensating transaction).
	eventTypes := []events.EventType{
		events.OrderCreated,
		events.OrderCancelled,
	}

	s.eventBus.Subscribe("payment-service", eventTypes, s.handleEvent)
}

// isProcessed provides idempotency.
func (s *Service) isProcessed(eventID string) bool {
	_, loaded := s.processedEvents.LoadOrStore(eventID, true)
	return loaded
}

// handleEvent processes incoming events.
func (s *Service) handleEvent(ctx context.Context, event *events.Event) error {
	if s.isProcessed(event.ID) {
		s.logger.Info("[PaymentService] Skipping duplicate event",
			"service", "payment",
			"event_id", event.ID,
		)
		return nil
	}

	s.logger.Info("[PaymentService] Received event",
		"service", "payment",
		"event_type", string(event.Type),
		"correlation_id", event.CorrelationID,
	)

	switch event.Type {
	case events.OrderCreated:
		return s.handleOrderCreated(ctx, event)
	case events.OrderCancelled:
		return s.handleOrderCancelled(ctx, event)
	default:
		return nil
	}
}

// handleOrderCreated processes payment when an order is created.
func (s *Service) handleOrderCreated(ctx context.Context, event *events.Event) error {
	var data events.OrderCreatedData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse order created data: %w", err)
	}

	order := data.Order
	s.logger.Info("[PaymentService] Processing payment",
		"service", "payment",
		"order_id", order.ID,
		"customer_id", order.CustomerID,
		"amount", order.TotalAmount,
	)

	// Create payment record
	payment := &models.Payment{
		ID:         events.GenerateID("PAY"),
		OrderID:    order.ID,
		CustomerID: order.CustomerID,
		Amount:     order.TotalAmount,
		Status:     models.PaymentStatusPending,
	}

	if err := s.repo.CreatePayment(ctx, payment); err != nil {
		return fmt.Errorf("failed to create payment: %w", err)
	}

	// Attempt to deduct from customer balance.
	// We capture the error in a distinct variable (deductErr) to avoid confusion
	// with the err returned by NewEvent below.
	if deductErr := s.repo.DeductBalance(ctx, order.CustomerID, order.TotalAmount); deductErr != nil {
		// Payment failed: publish failure event so the Order service can cancel the saga
		payment.Status = models.PaymentStatusFailed
		_ = s.repo.UpdatePaymentStatus(ctx, payment.ID, models.PaymentStatusFailed)

		s.logger.Warn("[PaymentService] Payment failed",
			"service", "payment",
			"order_id", order.ID,
			"error", deductErr,
		)

		failedEvent, err := events.NewEvent(
			events.PaymentFailed,
			order.ID,
			"payment",
			events.PaymentFailedData{
				OrderID:    order.ID,
				CustomerID: order.CustomerID,
				Reason:     deductErr.Error(),
			},
		)
		if err != nil {
			return fmt.Errorf("failed to create payment failed event: %w", err)
		}

		failedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
		return s.eventBus.Publish(ctx, failedEvent)
	}

	// Payment successful
	payment.Status = models.PaymentStatusCompleted
	if err := s.repo.UpdatePaymentStatus(ctx, payment.ID, models.PaymentStatusCompleted); err != nil {
		return fmt.Errorf("failed to update payment status: %w", err)
	}

	s.logger.Info("[PaymentService] Payment completed",
		"service", "payment",
		"order_id", order.ID,
		"payment_id", payment.ID,
	)

	// Publish payment completed event to trigger inventory reservation
	completedEvent, err := events.NewEvent(
		events.PaymentCompleted,
		order.ID,
		"payment",
		events.PaymentCompletedData{
			PaymentID:  payment.ID,
			OrderID:    order.ID,
			CustomerID: order.CustomerID,
			Amount:     order.TotalAmount,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create payment completed event: %w", err)
	}

	completedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
	return s.eventBus.Publish(ctx, completedEvent)
}

// handleOrderCancelled issues a refund as a compensating transaction.
// This method demonstrates the core principle of saga compensation: instead of
// "rolling back" the payment (which is impossible once money has moved), we
// execute a new forward action (a refund) that semantically reverses the effect.
func (s *Service) handleOrderCancelled(ctx context.Context, event *events.Event) error {
	var data events.OrderCancelledData
	if err := event.ParseData(&data); err != nil {
		return fmt.Errorf("failed to parse order cancelled data: %w", err)
	}

	// Find the payment for this order
	payment, err := s.repo.GetPaymentByOrderID(data.OrderID)
	if err != nil {
		// No payment found: nothing to refund.
		// This is normal when the saga failed before payment was attempted.
		s.logger.Info("[PaymentService] No payment found for cancelled order",
			"service", "payment",
			"order_id", data.OrderID,
		)
		return nil
	}

	// Only refund completed payments.
	// A failed or already-refunded payment needs no compensation.
	if payment.Status != models.PaymentStatusCompleted {
		s.logger.Info("[PaymentService] Payment not in completed status, skipping refund",
			"service", "payment",
			"payment_id", payment.ID,
			"status", string(payment.Status),
		)
		return nil
	}

	// Issue refund (compensating transaction)
	s.logger.Info("[PaymentService] Issuing refund",
		"service", "payment",
		"order_id", data.OrderID,
		"payment_id", payment.ID,
		"amount", payment.Amount,
	)

	// Credit the amount back to customer
	if err = s.repo.CreditBalance(ctx, payment.CustomerID, payment.Amount); err != nil {
		return fmt.Errorf("failed to credit balance: %w", err)
	}

	// Update payment status
	if err = s.repo.UpdatePaymentStatus(ctx, payment.ID, models.PaymentStatusRefunded); err != nil {
		return fmt.Errorf("failed to update payment status: %w", err)
	}

	s.logger.Info("[PaymentService] Refund completed",
		"service", "payment",
		"order_id", data.OrderID,
	)

	// Publish refund event so the Order service (and monitor) can track it
	refundedEvent, err := events.NewEvent(
		events.PaymentRefunded,
		data.OrderID,
		"payment",
		events.PaymentRefundedData{
			PaymentID:  payment.ID,
			OrderID:    data.OrderID,
			CustomerID: payment.CustomerID,
			Amount:     payment.Amount,
			Reason:     data.Reason,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create refund event: %w", err)
	}

	refundedEvent.WithCorrelation(event.CorrelationID).WithCausation(event.ID)
	return s.eventBus.Publish(ctx, refundedEvent)
}

// GetCustomer retrieves customer information.
func (s *Service) GetCustomer(id string) (*models.Customer, error) {
	return s.repo.GetCustomer(id)
}
