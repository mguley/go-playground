package services

import (
	"fmt"
	"time"
)

// TripBookingRequest represents a customer's request to book a complete trip.
// This is the input that initiates our saga, containing all the details
// needed for flight, hotel, and car reservations.
type TripBookingRequest struct {
	CustomerID    string `json:"customer_id"`
	CustomerEmail string `json:"customer_email"`

	// Flight details - these determine the travel dates that other
	// services need to align with
	FlightOrigin      string    `json:"flight_origin"`
	FlightDestination string    `json:"flight_destination"`
	FlightDate        time.Time `json:"flight_date"`
	FlightClass       string    `json:"flight_class"` // ECONOMY, BUSINESS, FIRST

	// Hotel details - typically aligned with the flight arrival date
	HotelCity     string    `json:"hotel_city"`
	HotelCheckIn  time.Time `json:"hotel_check_in"`
	HotelCheckOut time.Time `json:"hotel_check_out"`
	HotelRoomType string    `json:"hotel_room_type"` // STANDARD, DELUXE, SUITE

	// Car rental details - pickup typically at destination airport
	CarPickupCity string    `json:"car_pickup_city"`
	CarPickupDate time.Time `json:"car_pickup_date"`
	CarReturnDate time.Time `json:"car_return_date"`
	CarType       string    `json:"car_type"` // ECONOMY, COMPACT, SUV, LUXURY
}

// FlightReservation represents a reserved flight returned by the flight service.
// The Status field tracks the reservation lifecycle: RESERVED → CONFIRMED or CANCELLED.
type FlightReservation struct {
	ReservationID string    `json:"reservation_id"`
	FlightNumber  string    `json:"flight_number"`
	Origin        string    `json:"origin"`
	Destination   string    `json:"destination"`
	DepartureTime time.Time `json:"departure_time"`
	ArrivalTime   time.Time `json:"arrival_time"`
	SeatNumber    string    `json:"seat_number"`
	Class         string    `json:"class"`
	Price         float64   `json:"price"`
	Status        string    `json:"status"` // RESERVED, CONFIRMED, CANCELLED
	CustomerID    string    `json:"customer_id"`
}

// HotelReservation represents a reserved hotel room.
type HotelReservation struct {
	ReservationID string    `json:"reservation_id"`
	HotelName     string    `json:"hotel_name"`
	HotelAddress  string    `json:"hotel_address"`
	RoomNumber    string    `json:"room_number"`
	RoomType      string    `json:"room_type"`
	CheckInDate   time.Time `json:"check_in_date"`
	CheckOutDate  time.Time `json:"check_out_date"`
	PricePerNight float64   `json:"price_per_night"`
	TotalPrice    float64   `json:"total_price"`
	Status        string    `json:"status"` // RESERVED, CONFIRMED, CANCELLED
	CustomerID    string    `json:"customer_id"`
}

// CarReservation represents a reserved rental car.
type CarReservation struct {
	ReservationID  string    `json:"reservation_id"`
	CarModel       string    `json:"car_model"`
	CarType        string    `json:"car_type"`
	PickupLocation string    `json:"pickup_location"`
	ReturnLocation string    `json:"return_location"`
	PickupDate     time.Time `json:"pickup_date"`
	ReturnDate     time.Time `json:"return_date"`
	PricePerDay    float64   `json:"price_per_day"`
	TotalPrice     float64   `json:"total_price"`
	Status         string    `json:"status"` // RESERVED, CONFIRMED, CANCELLED
	CustomerID     string    `json:"customer_id"`
}

// TripBookingResult contains all reservations for a completed trip booking.
// This is returned when the saga completes successfully.
type TripBookingResult struct {
	BookingID         string             `json:"booking_id"`
	CustomerID        string             `json:"customer_id"`
	FlightReservation *FlightReservation `json:"flight_reservation,omitempty"`
	HotelReservation  *HotelReservation  `json:"hotel_reservation,omitempty"`
	CarReservation    *CarReservation    `json:"car_reservation,omitempty"`
	TotalPrice        float64            `json:"total_price"`
	Status            string             `json:"status"`
	CreatedAt         time.Time          `json:"created_at"`
}

// ServiceError represents an error from a service with additional context.
// The IsRetryable field is crucial for the saga to determine whether
// to retry the operation or immediately begin compensation.
type ServiceError struct {
	Service     string `json:"service"`
	Code        string `json:"code"`
	Message     string `json:"message"`
	IsRetryable bool   `json:"retryable"`
	Details     string `json:"details,omitempty"`
}

// Error implements the error interface.
func (e *ServiceError) Error() string {
	return fmt.Sprintf("%s service error [%s]: %s", e.Service, e.Code, e.Message)
}

// Retryable returns whether this error indicates a transient failure
// that might succeed if retried. The saga framework checks this to
// decide between retrying and compensating.
func (e *ServiceError) Retryable() bool {
	return e.IsRetryable
}

// NewServiceError creates a new service error with all fields populated.
func NewServiceError(service, code, message string, retryable bool) *ServiceError {
	return &ServiceError{
		Service:     service,
		Code:        code,
		Message:     message,
		IsRetryable: retryable,
	}
}
