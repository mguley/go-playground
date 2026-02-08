package services

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// CarService simulates a car rental reservation system.
type CarService struct {
	mu           sync.RWMutex
	reservations map[string]*CarReservation
	failureRate  float64
	latency      time.Duration
}

func NewCarService() *CarService {
	return &CarService{
		reservations: make(map[string]*CarReservation),
		failureRate:  0.0,
		latency:      50 * time.Millisecond,
	}
}

func (s *CarService) SetFailureRate(rate float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failureRate = rate
}

func (s *CarService) SetLatency(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.latency = d
}

var carModels = map[string][]string{
	"ECONOMY": {"Toyota Corolla", "Honda Civic", "Nissan Sentra"},
	"COMPACT": {"Mazda 3", "Volkswagen Golf", "Ford Focus"},
	"SUV":     {"Toyota RAV4", "Honda CR-V", "Ford Escape"},
	"LUXURY":  {"BMW 5 Series", "Mercedes E-Class", "Audi A6"},
}

func (s *CarService) ReserveCar(ctx context.Context, req *TripBookingRequest) (*CarReservation, error) {
	s.mu.RLock()
	latency := s.latency
	failureRate := s.failureRate
	s.mu.RUnlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(latency):
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if failureRate > 0 && rand.Float64() < failureRate {
		return nil, NewServiceError("car", "RANDOM_FAILURE",
			"simulated car service failure", true)
	}

	reservationID := fmt.Sprintf("CR-%d", time.Now().UnixNano())

	carType := req.CarType
	if carType == "" {
		carType = "ECONOMY"
	}

	models, ok := carModels[carType]
	if !ok {
		models = carModels["ECONOMY"]
	}
	carModel := models[rand.Intn(len(models))]

	basePricePerDay := 30.0 + rand.Float64()*20
	multiplier := 1.0
	switch carType {
	case "COMPACT":
		multiplier = 1.3
	case "SUV":
		multiplier = 1.8
	case "LUXURY":
		multiplier = 3.0
	}

	pricePerDay := basePricePerDay * multiplier
	days := int(req.CarReturnDate.Sub(req.CarPickupDate).Hours() / 24)
	if days < 1 {
		days = 1
	}

	reservation := &CarReservation{
		ReservationID:  reservationID,
		CarModel:       carModel,
		CarType:        carType,
		PickupLocation: fmt.Sprintf("%s Airport", req.CarPickupCity),
		ReturnLocation: fmt.Sprintf("%s Airport", req.CarPickupCity),
		PickupDate:     req.CarPickupDate,
		ReturnDate:     req.CarReturnDate,
		PricePerDay:    pricePerDay,
		TotalPrice:     pricePerDay * float64(days),
		Status:         "RESERVED",
		CustomerID:     req.CustomerID,
	}

	s.reservations[reservationID] = reservation
	return reservation, nil
}

// CancelCar cancels a car rental reservation (compensation action).
// Idempotent - safe to call multiple times.
func (s *CarService) CancelCar(ctx context.Context, reservationID string) error {
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
		return nil
	}
	if reservation.Status == "CANCELLED" {
		return nil
	}

	reservation.Status = "CANCELLED"
	return nil
}

func (s *CarService) GetReservation(ctx context.Context, reservationID string) (*CarReservation, error) {
	s.mu.RLock()
	latency := s.latency
	s.mu.RUnlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(latency / 2):
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	reservation, exists := s.reservations[reservationID]
	if !exists {
		return nil, NewServiceError("car", "NOT_FOUND",
			fmt.Sprintf("reservation %s not found", reservationID), false)
	}
	return reservation, nil
}
