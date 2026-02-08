package services

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// FlightService simulates an airline reservation system. In production,
// this would be a separate microservice with its own database. We
// simulate network latency and configurable failure rates for testing.
type FlightService struct {
	mu           sync.RWMutex
	reservations map[string]*FlightReservation

	// Configuration for simulating failures - essential for testing compensation
	failureRate float64       // Probability of random failure (0.0 to 1.0)
	latency     time.Duration // Simulated network latency
}

// NewFlightService creates a new flight service instance with default settings.
func NewFlightService() *FlightService {
	return &FlightService{
		reservations: make(map[string]*FlightReservation),
		failureRate:  0.0,
		latency:      50 * time.Millisecond,
	}
}

// SetFailureRate configures the probability of simulated failures.
// A rate of 1.0 means every request will fail; 0.0 means no simulated failures.
func (s *FlightService) SetFailureRate(rate float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failureRate = rate
}

// SetLatency configures simulated network latency for all operations.
func (s *FlightService) SetLatency(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.latency = d
}

// ReserveFlight creates a new flight reservation. In a real system this
// is a two-phase operation: first reserve (hold the seat), then confirm
// (charge the customer). Reserved flights are held for a limited time
// before being automatically released.
func (s *FlightService) ReserveFlight(ctx context.Context, req *TripBookingRequest) (*FlightReservation, error) {
	// Read config under RLock, then release before sleeping
	s.mu.RLock()
	latency := s.latency
	failureRate := s.failureRate
	s.mu.RUnlock()

	// Simulate network latency, respecting context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(latency):
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Simulate random failures for testing compensation
	if failureRate > 0 && rand.Float64() < failureRate {
		return nil, NewServiceError("flight", "RANDOM_FAILURE",
			"simulated flight service failure", true)
	}

	reservationID := fmt.Sprintf("FL-%d", time.Now().UnixNano())

	// Simulate finding an available flight
	flightNumber := fmt.Sprintf("AA%d", 100+rand.Intn(900))
	departureTime := req.FlightDate.Add(time.Hour * time.Duration(6+rand.Intn(12)))
	arrivalTime := departureTime.Add(time.Hour * time.Duration(2+rand.Intn(6)))

	// Calculate price based on class
	basePrice := 200.0 + rand.Float64()*300
	multiplier := 1.0
	switch req.FlightClass {
	case "BUSINESS":
		multiplier = 2.5
	case "FIRST":
		multiplier = 4.0
	}

	reservation := &FlightReservation{
		ReservationID: reservationID,
		FlightNumber:  flightNumber,
		Origin:        req.FlightOrigin,
		Destination:   req.FlightDestination,
		DepartureTime: departureTime,
		ArrivalTime:   arrivalTime,
		SeatNumber:    fmt.Sprintf("%d%c", rand.Intn(30)+1, 'A'+rune(rand.Intn(6))),
		Class:         req.FlightClass,
		Price:         basePrice * multiplier,
		Status:        "RESERVED",
		CustomerID:    req.CustomerID,
	}

	s.reservations[reservationID] = reservation
	return reservation, nil
}

// CancelFlight cancels a flight reservation. This is the compensation
// action for flight booking. It MUST be idempotent - calling it multiple
// times has the same effect as calling it once. If the reservation
// doesn't exist or is already cancelled, we consider it a success
// because the end state (no active reservation) is what we want.
func (s *FlightService) CancelFlight(ctx context.Context, reservationID string) error {
	s.mu.RLock()
	latency := s.latency
	s.mu.RUnlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(latency):
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	reservation, exists := s.reservations[reservationID]
	if !exists {
		// Idempotent: if reservation doesn't exist, the desired state
		// (no active reservation) is already achieved
		return nil
	}

	if reservation.Status == "CANCELLED" {
		// Idempotent: already cancelled
		return nil
	}

	reservation.Status = "CANCELLED"
	return nil
}

// GetReservation retrieves a reservation by ID for status checking.
func (s *FlightService) GetReservation(ctx context.Context, reservationID string) (*FlightReservation, error) {
	s.mu.RLock()
	latency := s.latency
	s.mu.RUnlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(latency / 2): // Reads are faster
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	reservation, exists := s.reservations[reservationID]
	if !exists {
		return nil, NewServiceError("flight", "NOT_FOUND",
			fmt.Sprintf("reservation %s not found", reservationID), false)
	}

	return reservation, nil
}
