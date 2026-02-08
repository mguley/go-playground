package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"sagaorchestration/internal/handlers"
	"sagaorchestration/internal/middleware"
	"sagaorchestration/pkg/orchestrator"
	"sagaorchestration/pkg/services"
	"sagaorchestration/pkg/store"
)

func main() {
	log.Println("Starting Trip Booking Saga Server...")

	// Initialize the three services that participate in our saga.
	// In production, these would be separate microservices.
	flightService := services.NewFlightService()
	hotelService := services.NewHotelService()
	carService := services.NewCarService()

	// Initialize saga store - using in-memory for this demo.
	// Production would use PostgreSQL, Redis, etc.
	sagaStore := store.NewMemoryStore()

	// Initialize the orchestrator
	orch := orchestrator.NewOrchestrator(sagaStore)

	// Configure event handler for observability
	orch.EventHandler = func(event orchestrator.SagaEvent) {
		log.Printf("[SAGA EVENT] Type=%s SagaID=%s Step=%s Status=%s Error=%s",
			event.Type, event.SagaID, event.StepName, event.Status, event.Error)
	}

	// Build and register the trip booking saga definition
	sagaBuilder := orchestrator.NewTripBookingSagaBuilder(flightService, hotelService, carService)
	if err := orch.RegisterDefinition(sagaBuilder.Build()); err != nil {
		log.Fatalf("Failed to register saga definition: %v", err)
	}

	// Initialize HTTP handlers and routes
	handler := handlers.NewHandler(orch)
	// Initialize metrics collector
	metrics := middleware.NewMetrics()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", handler.HealthCheck)
	mux.HandleFunc("/api/bookings", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			handler.ListBookings(w, r)
		case http.MethodPost:
			handler.BookTrip(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/api/bookings/", handler.GetBookingStatus)
	mux.Handle("/metrics", metrics.Handler())

	wrappedHandler := metrics.Middleware(middleware.Logger(mux))

	server := &http.Server{
		Addr:         ":8080",
		Handler:      wrappedHandler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Printf("Server listening on %s", server.Addr)
		log.Println("Endpoints:")
		log.Println("  GET  /health              - Health check")
		log.Println("  POST /api/bookings        - Create a trip booking")
		log.Println("  GET  /api/bookings        - List all bookings")
		log.Println("  GET  /api/bookings/{id}   - Get booking status")
		log.Println("  GET  /metrics             - Request metrics")

		if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Graceful shutdown on interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}
	log.Println("Server stopped")
}
