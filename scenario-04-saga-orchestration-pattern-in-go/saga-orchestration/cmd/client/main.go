package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

const baseURL = "http://localhost:8080"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "book":
		bookTrip()
	case "status":
		if len(os.Args) < 3 {
			log.Fatal("Usage: client status <booking-id>")
		}
		getStatus(os.Args[2])
	case "list":
		listBookings()
	case "health":
		checkHealth()
	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Usage: client <command> [args]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  book              Create a new trip booking")
	fmt.Println("  status <id>       Get booking status")
	fmt.Println("  list              List all bookings")
	fmt.Println("  health            Check server health")
}

func bookTrip() {
	now := time.Now()

	booking := map[string]any{
		"customer_id":        "CLI-USER-001",
		"customer_email":     "user@example.com",
		"flight_origin":      "SFO",
		"flight_destination": "JFK",
		"flight_date":        now.Add(24 * time.Hour).Format(time.RFC3339),
		"flight_class":       "BUSINESS",
		"hotel_city":         "New York",
		"hotel_check_in":     now.Add(24 * time.Hour).Format(time.RFC3339),
		"hotel_check_out":    now.Add(96 * time.Hour).Format(time.RFC3339),
		"hotel_room_type":    "DELUXE",
		"car_pickup_city":    "New York",
		"car_pickup_date":    now.Add(24 * time.Hour).Format(time.RFC3339),
		"car_return_date":    now.Add(96 * time.Hour).Format(time.RFC3339),
		"car_type":           "SUV",
	}

	body, _ := json.Marshal(booking)

	resp, err := http.Post(baseURL+"/api/bookings", "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)

	fmt.Printf("Status: %d\n", resp.StatusCode)
	prettyPrintJSON(respBody)

	// If accepted, poll for completion
	if resp.StatusCode == http.StatusAccepted {
		var result map[string]any
		_ = json.Unmarshal(respBody, &result)

		if bookingID, ok := result["booking_id"].(string); ok {
			fmt.Println("\nPolling for completion...")
			pollStatus(bookingID)
		}
	}
}

func pollStatus(bookingID string) {
	for i := 0; i < 30; i++ {
		time.Sleep(500 * time.Millisecond)

		resp, err := http.Get(baseURL + "/api/bookings/" + bookingID)
		if err != nil {
			continue
		}

		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		var status map[string]any
		_ = json.Unmarshal(body, &status)

		statusStr, _ := status["status"].(string)
		fmt.Printf("  Status: %s\n", statusStr)

		if statusStr == "COMPLETED" || statusStr == "COMPENSATED" || statusStr == "FAILED" {
			fmt.Println("\nFinal result:")
			prettyPrintJSON(body)
			return
		}
	}
	fmt.Println("Timeout waiting for completion")
}

func getStatus(bookingID string) {
	resp, err := http.Get(baseURL + "/api/bookings/" + bookingID)
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Status: %d\n", resp.StatusCode)
	prettyPrintJSON(body)
}

func listBookings() {
	resp, err := http.Get(baseURL + "/api/bookings")
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Status: %d\n", resp.StatusCode)
	prettyPrintJSON(body)
}

func checkHealth() {
	resp, err := http.Get(baseURL + "/health")
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Status: %d\n", resp.StatusCode)
	prettyPrintJSON(body)
}

func prettyPrintJSON(data []byte) {
	var parsed any
	if err := json.Unmarshal(data, &parsed); err == nil {
		output, _ := json.MarshalIndent(parsed, "", "  ")
		fmt.Println(string(output))
	} else {
		fmt.Println(string(data))
	}
}
