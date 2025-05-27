package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

func captureHeapProfile(filename string) error {
	resp, err := http.Get("http://localhost:8080/debug/pprof/heap?seconds=25")
	if err != nil {
		return err
	}
	defer func(Body io.ReadCloser) {
		if err = Body.Close(); err != nil {
			log.Fatal(err)
		}
	}(resp.Body)

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func(file *os.File) {
		if err = file.Close(); err != nil {
			log.Fatal(err)
		}
	}(file)

	_, err = file.ReadFrom(resp.Body)
	return err
}

func main() {
	// Capture initial heap profile
	fmt.Println("Capturing initial heap profile...")
	if err := captureHeapProfile("heap1.prof"); err != nil {
		log.Fatal(err)
	}

	// Generate load
	fmt.Println("Generating load for 30 seconds...")
	if _, err := http.Get("http://localhost:8080/load"); err != nil {
		return
	}

	// Capture second heap profile
	fmt.Println("Capturing second heap profile...")
	if err := captureHeapProfile("heap2.prof"); err != nil {
		log.Fatal(err)
	}

	fmt.Println("\nCompare profiles with:")
	fmt.Println("go tool pprof -base=heap1.prof heap2.prof")
	fmt.Println("\nThen use 'top' command to see memory growth")
}
