package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"sagaorchestration/pkg/saga"
	"sagaorchestration/pkg/services"
)

// TripBookingSagaBuilder creates a saga definition for the trip booking workflow.
type TripBookingSagaBuilder struct {
	flightService *services.FlightService
	hotelService  *services.HotelService
	carService    *services.CarService
}

func NewTripBookingSagaBuilder(
	flightSvc *services.FlightService,
	hotelSvc *services.HotelService,
	carSvc *services.CarService,
) *TripBookingSagaBuilder {
	return &TripBookingSagaBuilder{
		flightService: flightSvc,
		hotelService:  hotelSvc,
		carService:    carSvc,
	}
}

// Build creates the saga definition. Steps are ordered by resource
// scarcity and dependency: flights first (most constrained and set
// travel dates), then hotels (depend on flight dates), then cars
// (most flexible).
func (b *TripBookingSagaBuilder) Build() *saga.Definition {
	def := saga.NewDefinition("TripBooking").
		WithDescription("Books a complete trip including flight, hotel, and car rental")

	// Step 1: Reserve Flight
	flightStep := saga.NewStep("reserve-flight", "Reserve Flight", b.reserveFlight).
		WithCompensation(b.cancelFlight).
		WithRetry(3, 200*time.Millisecond).
		WithTimeout(30 * time.Second)
	def.AddStep(flightStep)

	// Step 2: Reserve Hotel
	hotelStep := saga.NewStep("reserve-hotel", "Reserve Hotel", b.reserveHotel).
		WithCompensation(b.cancelHotel).
		WithRetry(3, 200*time.Millisecond).
		WithTimeout(30 * time.Second)
	def.AddStep(hotelStep)

	// Step 3: Reserve Car - if this fails, both hotel and flight are compensated
	carStep := saga.NewStep("reserve-car", "Reserve Car", b.reserveCar).
		WithCompensation(b.cancelCar).
		WithRetry(3, 200*time.Millisecond).
		WithTimeout(30 * time.Second)
	def.AddStep(carStep)

	return def
}

// reserveFlight extracts the booking request from saga state and calls
// the flight service. Results are stored back into saga data so that
// the compensation function (and subsequent steps) can access them.
func (b *TripBookingSagaBuilder) reserveFlight(ctx context.Context, state *saga.SagaState) (map[string]any, error) {
	var req services.TripBookingRequest
	if err := state.GetDataAs("booking_request", &req); err != nil {
		return nil, fmt.Errorf("failed to get booking request: %w", err)
	}

	reservation, err := b.flightService.ReserveFlight(ctx, &req)
	if err != nil {
		return nil, err
	}

	// Convert to map for saga data storage via JSON round-trip
	reservationJSON, err := json.Marshal(reservation)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal flight reservation: %w", err)
	}
	var result map[string]any
	if err = json.Unmarshal(reservationJSON, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal flight reservation: %w", err)
	}

	return map[string]any{
		"flight_reservation":    result,
		"flight_reservation_id": reservation.ReservationID,
	}, nil
}

// cancelFlight compensates a flight reservation. It retrieves the
// reservation ID from saga data and cancels it. Safe to call even
// if no reservation was created (returns nil).
func (b *TripBookingSagaBuilder) cancelFlight(ctx context.Context, state *saga.SagaState) error {
	reservationID, ok := state.GetData("flight_reservation_id")
	if !ok {
		return nil // No reservation to cancel
	}

	idStr, ok := reservationID.(string)
	if !ok {
		return fmt.Errorf("invalid flight_reservation_id type: %T", reservationID)
	}

	return b.flightService.CancelFlight(ctx, idStr)
}

func (b *TripBookingSagaBuilder) reserveHotel(ctx context.Context, state *saga.SagaState) (map[string]any, error) {
	var req services.TripBookingRequest
	if err := state.GetDataAs("booking_request", &req); err != nil {
		return nil, fmt.Errorf("failed to get booking request: %w", err)
	}

	reservation, err := b.hotelService.ReserveHotel(ctx, &req)
	if err != nil {
		return nil, err
	}

	reservationJSON, err := json.Marshal(reservation)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal hotel reservation: %w", err)
	}
	var result map[string]any
	if err = json.Unmarshal(reservationJSON, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal hotel reservation: %w", err)
	}

	return map[string]any{
		"hotel_reservation":    result,
		"hotel_reservation_id": reservation.ReservationID,
	}, nil
}

func (b *TripBookingSagaBuilder) cancelHotel(ctx context.Context, state *saga.SagaState) error {
	reservationID, ok := state.GetData("hotel_reservation_id")
	if !ok {
		return nil
	}

	idStr, ok := reservationID.(string)
	if !ok {
		return fmt.Errorf("invalid hotel_reservation_id type: %T", reservationID)
	}

	return b.hotelService.CancelHotel(ctx, idStr)
}

func (b *TripBookingSagaBuilder) reserveCar(ctx context.Context, state *saga.SagaState) (map[string]any, error) {
	var req services.TripBookingRequest
	if err := state.GetDataAs("booking_request", &req); err != nil {
		return nil, fmt.Errorf("failed to get booking request: %w", err)
	}

	reservation, err := b.carService.ReserveCar(ctx, &req)
	if err != nil {
		return nil, err
	}

	reservationJSON, err := json.Marshal(reservation)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal car reservation: %w", err)
	}
	var result map[string]any
	if err = json.Unmarshal(reservationJSON, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal car reservation: %w", err)
	}

	return map[string]any{
		"car_reservation":    result,
		"car_reservation_id": reservation.ReservationID,
	}, nil
}

func (b *TripBookingSagaBuilder) cancelCar(ctx context.Context, state *saga.SagaState) error {
	reservationID, ok := state.GetData("car_reservation_id")
	if !ok {
		return nil
	}

	idStr, ok := reservationID.(string)
	if !ok {
		return fmt.Errorf("invalid car_reservation_id type: %T", reservationID)
	}

	return b.carService.CancelCar(ctx, idStr)
}
