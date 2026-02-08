package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"sagaorchestration/pkg/orchestrator"
	"sagaorchestration/pkg/saga"
	"sagaorchestration/pkg/services"
	"sagaorchestration/pkg/store"
)

// Handler handles HTTP requests for the saga API.
type Handler struct {
	orchestrator *orchestrator.Orchestrator
}

func NewHandler(orch *orchestrator.Orchestrator) *Handler {
	return &Handler{orchestrator: orch}
}

// BookTripRequest is the API request for booking a trip.
type BookTripRequest struct {
	CustomerID    string `json:"customer_id"`
	CustomerEmail string `json:"customer_email"`

	FlightOrigin      string `json:"flight_origin"`
	FlightDestination string `json:"flight_destination"`
	FlightDate        string `json:"flight_date"` // RFC3339 format
	FlightClass       string `json:"flight_class"`

	HotelCity     string `json:"hotel_city"`
	HotelCheckIn  string `json:"hotel_check_in"`  // RFC3339 format
	HotelCheckOut string `json:"hotel_check_out"` // RFC3339 format
	HotelRoomType string `json:"hotel_room_type"`

	CarPickupCity string `json:"car_pickup_city"`
	CarPickupDate string `json:"car_pickup_date"` // RFC3339 format
	CarReturnDate string `json:"car_return_date"` // RFC3339 format
	CarType       string `json:"car_type"`
}

type BookTripResponse struct {
	BookingID string `json:"booking_id"`
	Status    string `json:"status"`
	Message   string `json:"message"`
}

type SagaStatusResponse struct {
	ID          string               `json:"id"`
	Name        string               `json:"name"`
	Status      string               `json:"status"`
	CurrentStep int                  `json:"current_step"`
	TotalSteps  int                  `json:"total_steps"`
	Steps       []StepStatusResponse `json:"steps"`
	Data        map[string]any       `json:"data,omitempty"`
	Error       string               `json:"error,omitempty"`
	CreatedAt   string               `json:"created_at"`
	UpdatedAt   string               `json:"updated_at"`
	CompletedAt string               `json:"completed_at,omitempty"`
}

type StepStatusResponse struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Status     string `json:"status"`
	Error      string `json:"error,omitempty"`
	RetryCount int    `json:"retry_count"`
}

// BookTrip handles POST /api/bookings - creates a new trip booking.
func (h *Handler) BookTrip(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req BookTripRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body: "+err.Error())
		return
	}

	if req.CustomerID == "" {
		respondError(w, http.StatusBadRequest, "customer_id is required")
		return
	}

	// Parse all dates - they must be valid RFC3339 format
	flightDate, err := time.Parse(time.RFC3339, req.FlightDate)
	if err != nil {
		respondError(w, http.StatusBadRequest, "Invalid flight_date format, expected RFC3339")
		return
	}

	hotelCheckIn, err := time.Parse(time.RFC3339, req.HotelCheckIn)
	if err != nil {
		respondError(w, http.StatusBadRequest, "Invalid hotel_check_in format, expected RFC3339")
		return
	}

	hotelCheckOut, err := time.Parse(time.RFC3339, req.HotelCheckOut)
	if err != nil {
		respondError(w, http.StatusBadRequest, "Invalid hotel_check_out format, expected RFC3339")
		return
	}

	carPickupDate, err := time.Parse(time.RFC3339, req.CarPickupDate)
	if err != nil {
		respondError(w, http.StatusBadRequest, "Invalid car_pickup_date format, expected RFC3339")
		return
	}

	carReturnDate, err := time.Parse(time.RFC3339, req.CarReturnDate)
	if err != nil {
		respondError(w, http.StatusBadRequest, "Invalid car_return_date format, expected RFC3339")
		return
	}

	bookingReq := services.TripBookingRequest{
		CustomerID:        req.CustomerID,
		CustomerEmail:     req.CustomerEmail,
		FlightOrigin:      req.FlightOrigin,
		FlightDestination: req.FlightDestination,
		FlightDate:        flightDate,
		FlightClass:       req.FlightClass,
		HotelCity:         req.HotelCity,
		HotelCheckIn:      hotelCheckIn,
		HotelCheckOut:     hotelCheckOut,
		HotelRoomType:     req.HotelRoomType,
		CarPickupCity:     req.CarPickupCity,
		CarPickupDate:     carPickupDate,
		CarReturnDate:     carReturnDate,
		CarType:           req.CarType,
	}

	// Start the saga - returns immediately while execution runs async
	sagaID, err := h.orchestrator.StartSaga(r.Context(), "TripBooking", map[string]any{
		"booking_request": bookingReq,
	})
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to start booking: "+err.Error())
		return
	}

	respondJSON(w, http.StatusAccepted, BookTripResponse{
		BookingID: string(sagaID),
		Status:    "PROCESSING",
		Message:   "Trip booking initiated. Poll the status endpoint for updates.",
	})
}

// GetBookingStatus handles GET /api/bookings/{id}.
func (h *Handler) GetBookingStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := r.URL.Path
	bookingID := strings.TrimPrefix(path, "/api/bookings/")
	if bookingID == "" || bookingID == path {
		respondError(w, http.StatusBadRequest, "Booking ID is required")
		return
	}

	state, err := h.orchestrator.GetSagaState(r.Context(), saga.SagaID(bookingID))
	if err != nil {
		respondError(w, http.StatusNotFound, "Booking not found")
		return
	}

	respondJSON(w, http.StatusOK, sagaStateToResponse(state))
}

// ListBookings handles GET /api/bookings.
func (h *Handler) ListBookings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	filter := store.SagaFilter{Name: "TripBooking", Limit: 50}
	if status := r.URL.Query().Get("status"); status != "" {
		filter.Status = saga.SagaStatus(status)
	}

	sagas, err := h.orchestrator.ListSagas(r.Context(), filter)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to list bookings")
		return
	}

	responses := make([]SagaStatusResponse, len(sagas))
	for i, s := range sagas {
		responses[i] = sagaStateToResponse(s)
	}

	respondJSON(w, http.StatusOK, map[string]any{
		"bookings": responses,
		"total":    len(responses),
	})
}

// HealthCheck handles GET /health.
func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, http.StatusOK, map[string]string{
		"status": "healthy",
		"time":   time.Now().Format(time.RFC3339),
	})
}

func sagaStateToResponse(state *saga.SagaState) SagaStatusResponse {
	steps := make([]StepStatusResponse, len(state.Steps))
	for i, step := range state.Steps {
		steps[i] = StepStatusResponse{
			ID:         string(step.ID),
			Name:       step.Name,
			Status:     string(step.Status),
			Error:      step.Error,
			RetryCount: step.RetryCount,
		}
	}

	completedAt := ""
	if state.CompletedAt != nil {
		completedAt = state.CompletedAt.Format(time.RFC3339)
	}

	return SagaStatusResponse{
		ID:          string(state.ID),
		Name:        state.Name,
		Status:      string(state.Status),
		CurrentStep: state.CurrentStep,
		TotalSteps:  len(state.Steps),
		Steps:       steps,
		Data:        state.Data,
		Error:       state.Error,
		CreatedAt:   state.CreatedAt.Format(time.RFC3339),
		UpdatedAt:   state.UpdatedAt.Format(time.RFC3339),
		CompletedAt: completedAt,
	}
}

func respondJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func respondError(w http.ResponseWriter, status int, message string) {
	respondJSON(w, status, map[string]string{"error": message})
}
