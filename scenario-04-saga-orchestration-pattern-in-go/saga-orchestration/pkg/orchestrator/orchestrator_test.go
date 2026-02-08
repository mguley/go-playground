package orchestrator

import (
	"context"
	"fmt"
	"testing"
	"time"

	"sagaorchestration/pkg/saga"
	"sagaorchestration/pkg/services"
	"sagaorchestration/pkg/store"
)

// createTestOrchestrator sets up a complete orchestrator for testing.
func createTestOrchestrator(t *testing.T) (*Orchestrator, *services.FlightService, *services.HotelService, *services.CarService) {
	t.Helper()

	flightSvc := services.NewFlightService()
	hotelSvc := services.NewHotelService()
	carSvc := services.NewCarService()
	sagaStore := store.NewMemoryStore()
	orch := NewOrchestrator(sagaStore)

	builder := NewTripBookingSagaBuilder(flightSvc, hotelSvc, carSvc)
	if err := orch.RegisterDefinition(builder.Build()); err != nil {
		t.Fatalf("Failed to register saga: %v", err)
	}

	return orch, flightSvc, hotelSvc, carSvc
}

// createTestBookingRequest creates a realistic test booking request.
func createTestBookingRequest() services.TripBookingRequest {
	now := time.Now()
	return services.TripBookingRequest{
		CustomerID:        "test-customer-123",
		CustomerEmail:     "test@example.com",
		FlightOrigin:      "NYC",
		FlightDestination: "LAX",
		FlightDate:        now.Add(24 * time.Hour),
		FlightClass:       "ECONOMY",
		HotelCity:         "Los Angeles",
		HotelCheckIn:      now.Add(24 * time.Hour),
		HotelCheckOut:     now.Add(72 * time.Hour),
		HotelRoomType:     "STANDARD",
		CarPickupCity:     "Los Angeles",
		CarPickupDate:     now.Add(24 * time.Hour),
		CarReturnDate:     now.Add(72 * time.Hour),
		CarType:           "ECONOMY",
	}
}

// waitForSagaCompletion polls until a saga reaches a terminal state.
func waitForSagaCompletion(t *testing.T, ctx context.Context, orch *Orchestrator, sagaID saga.SagaID) *saga.SagaState {
	t.Helper()

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			t.Fatalf("Timed out waiting for saga %s to complete", sagaID)
			return nil
		case <-ticker.C:
			state, err := orch.GetSagaState(ctx, sagaID)
			if err != nil {
				t.Fatalf("Failed to get saga state: %v", err)
			}
			if state.IsTerminal() {
				return state
			}
		}
	}
}

// TestSuccessfulTripBooking tests the happy path where all services succeed.
func TestSuccessfulTripBooking(t *testing.T) {
	orch, _, _, _ := createTestOrchestrator(t)
	bookingReq := createTestBookingRequest()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	sagaID, err := orch.StartSaga(ctx, "TripBooking", map[string]any{
		"booking_request": bookingReq,
	})
	if err != nil {
		t.Fatalf("Failed to start saga: %v", err)
	}

	state := waitForSagaCompletion(t, ctx, orch, sagaID)

	if state.Status != saga.StatusCompleted {
		t.Errorf("Expected status COMPLETED, got %s. Error: %s", state.Status, state.Error)
	}

	// Verify all steps completed
	for i, step := range state.Steps {
		if step.Status != saga.StepCompleted {
			t.Errorf("Step %d (%s) status: %s, expected COMPLETED", i, step.Name, step.Status)
		}
	}

	// Verify all reservation IDs were created
	for _, key := range []string{"flight_reservation_id", "hotel_reservation_id", "car_reservation_id"} {
		if _, ok := state.GetData(key); !ok {
			t.Errorf("%s not found in saga data", key)
		}
	}

	t.Logf("Saga completed successfully with ID: %s", sagaID)
}

// TestFlightServiceFailure tests compensation when the flight service fails.
// Since flight is the first step, no compensation is needed - there's
// nothing to undo.
func TestFlightServiceFailure(t *testing.T) {
	orch, flightSvc, _, _ := createTestOrchestrator(t)
	flightSvc.SetFailureRate(1.0) // 100% failure rate

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	sagaID, err := orch.StartSaga(ctx, "TripBooking", map[string]any{
		"booking_request": createTestBookingRequest(),
	})
	if err != nil {
		t.Fatalf("Failed to start saga: %v", err)
	}

	state := waitForSagaCompletion(t, ctx, orch, sagaID)

	if state.Status != saga.StatusCompensated {
		t.Errorf("Expected status COMPENSATED, got %s. Error: %s", state.Status, state.Error)
	}
	if state.Steps[0].Status != saga.StepFailed {
		t.Errorf("Expected first step FAILED, got %s", state.Steps[0].Status)
	}

	t.Log("Flight failure handled correctly - saga compensated (nothing to undo)")
}

// TestHotelServiceFailure tests compensation when hotel fails after flight succeeds.
// The flight reservation should be cancelled.
func TestHotelServiceFailure(t *testing.T) {
	orch, _, hotelSvc, _ := createTestOrchestrator(t)
	hotelSvc.SetFailureRate(1.0)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	sagaID, err := orch.StartSaga(ctx, "TripBooking", map[string]any{
		"booking_request": createTestBookingRequest(),
	})
	if err != nil {
		t.Fatalf("Failed to start saga: %v", err)
	}

	state := waitForSagaCompletion(t, ctx, orch, sagaID)

	if state.Status != saga.StatusCompensated {
		t.Errorf("Expected status COMPENSATED, got %s. Error: %s", state.Status, state.Error)
	}
	if state.Steps[0].Status != saga.StepCompensated {
		t.Errorf("Expected flight step COMPENSATED, got %s", state.Steps[0].Status)
	}
	if state.Steps[1].Status != saga.StepFailed {
		t.Errorf("Expected hotel step FAILED, got %s", state.Steps[1].Status)
	}

	t.Log("Hotel failure handled - flight reservation compensated")
}

// TestCarServiceFailure tests compensation when car fails after flight and hotel succeed.
// Both hotel and flight reservations should be cancelled in reverse order.
func TestCarServiceFailure(t *testing.T) {
	orch, _, _, carSvc := createTestOrchestrator(t)
	carSvc.SetFailureRate(1.0)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	sagaID, err := orch.StartSaga(ctx, "TripBooking", map[string]any{
		"booking_request": createTestBookingRequest(),
	})
	if err != nil {
		t.Fatalf("Failed to start saga: %v", err)
	}

	state := waitForSagaCompletion(t, ctx, orch, sagaID)

	if state.Status != saga.StatusCompensated {
		t.Errorf("Expected status COMPENSATED, got %s. Error: %s", state.Status, state.Error)
	}
	if state.Steps[0].Status != saga.StepCompensated {
		t.Errorf("Expected flight step COMPENSATED, got %s", state.Steps[0].Status)
	}
	if state.Steps[1].Status != saga.StepCompensated {
		t.Errorf("Expected hotel step COMPENSATED, got %s", state.Steps[1].Status)
	}
	if state.Steps[2].Status != saga.StepFailed {
		t.Errorf("Expected car step FAILED, got %s", state.Steps[2].Status)
	}

	t.Log("Car failure handled - all prior reservations compensated")
}

// TestConcurrentSagas verifies thread-safety by running multiple sagas at once.
func TestConcurrentSagas(t *testing.T) {
	orch, _, _, _ := createTestOrchestrator(t)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	numSagas := 10
	sagaIDs := make([]saga.SagaID, numSagas)

	for i := 0; i < numSagas; i++ {
		req := createTestBookingRequest()
		req.CustomerID = fmt.Sprintf("customer-%d", i)

		sagaID, err := orch.StartSaga(ctx, "TripBooking", map[string]any{
			"booking_request": req,
		})
		if err != nil {
			t.Fatalf("Failed to start saga %d: %v", i, err)
		}
		sagaIDs[i] = sagaID
	}

	completed := 0
	for _, sagaID := range sagaIDs {
		state := waitForSagaCompletion(t, ctx, orch, sagaID)
		if state.Status == saga.StatusCompleted {
			completed++
		}
	}

	if completed != numSagas {
		t.Errorf("Expected %d completed sagas, got %d", numSagas, completed)
	}

	t.Logf("Successfully ran %d concurrent sagas", completed)
}
