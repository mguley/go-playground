package services

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// HotelService simulates a hotel reservation system.
type HotelService struct {
	mu           sync.RWMutex
	reservations map[string]*HotelReservation
	failureRate  float64
	latency      time.Duration
}

func NewHotelService() *HotelService {
	return &HotelService{
		reservations: make(map[string]*HotelReservation),
		failureRate:  0.0,
		latency:      50 * time.Millisecond,
	}
}

func (s *HotelService) SetFailureRate(rate float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failureRate = rate
}

func (s *HotelService) SetLatency(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.latency = d
}

var hotelNames = []string{
	"Grand Plaza Hotel",
	"Seaside Resort & Spa",
	"Metropolitan Inn",
	"Sunset Beach Hotel",
	"Mountain View Lodge",
}

func (s *HotelService) ReserveHotel(ctx context.Context, req *TripBookingRequest) (*HotelReservation, error) {
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
		return nil, NewServiceError("hotel", "RANDOM_FAILURE",
			"simulated hotel service failure", true)
	}

	reservationID := fmt.Sprintf("HT-%d", time.Now().UnixNano())
	hotelName := hotelNames[rand.Intn(len(hotelNames))]

	basePricePerNight := 100.0 + rand.Float64()*100
	multiplier := 1.0
	switch req.HotelRoomType {
	case "DELUXE":
		multiplier = 1.5
	case "SUITE":
		multiplier = 2.5
	}

	pricePerNight := basePricePerNight * multiplier
	nights := int(req.HotelCheckOut.Sub(req.HotelCheckIn).Hours() / 24)
	if nights < 1 {
		nights = 1
	}

	reservation := &HotelReservation{
		ReservationID: reservationID,
		HotelName:     hotelName,
		HotelAddress:  fmt.Sprintf("123 Main St, %s", req.HotelCity),
		RoomNumber:    fmt.Sprintf("%d", 100+rand.Intn(900)),
		RoomType:      req.HotelRoomType,
		CheckInDate:   req.HotelCheckIn,
		CheckOutDate:  req.HotelCheckOut,
		PricePerNight: pricePerNight,
		TotalPrice:    pricePerNight * float64(nights),
		Status:        "RESERVED",
		CustomerID:    req.CustomerID,
	}

	s.reservations[reservationID] = reservation
	return reservation, nil
}

// CancelHotel cancels a hotel reservation (compensation action).
// Idempotent - safe to call multiple times.
func (s *HotelService) CancelHotel(ctx context.Context, reservationID string) error {
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
		return nil // Idempotent: nothing to cancel
	}
	if reservation.Status == "CANCELLED" {
		return nil // Idempotent: already cancelled
	}

	reservation.Status = "CANCELLED"
	return nil
}

func (s *HotelService) GetReservation(ctx context.Context, reservationID string) (*HotelReservation, error) {
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
		return nil, NewServiceError("hotel", "NOT_FOUND",
			fmt.Sprintf("reservation %s not found", reservationID), false)
	}
	return reservation, nil
}
