package main

import (
	"flag"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"strings"
)

var (
	profileDir  = flag.String("dir", "profiles", "Profile directory")
	profileType = flag.String("type", "cpu", "Profile type to view")
	compare     = flag.Bool("compare", false, "Compare two profiles")
)

func main() {
	flag.Parse()

	profiles, err := findProfiles(*profileDir, *profileType)
	if err != nil {
		log.Fatal(err)
	}

	if len(profiles) == 0 {
		log.Fatal("No profiles found")
	}

	switch {
	case *compare && len(profiles) >= 2:
		// Check if this profile type supports comparison
		if *profileType == "trace" {
			log.Fatal("Trace profiles cannot be compared. Use individual trace viewing instead.")
		}
		// Compare the two most recent profiles
		compareProfiles(profiles[len(profiles)-2], profiles[len(profiles)-1])
	default:
		// View the most recent profile
		viewProfile(profiles[len(profiles)-1])
	}
}

func findProfiles(dir, profileType string) ([]string, error) {
	pattern := filepath.Join(dir, fmt.Sprintf("%s_*.prof", profileType))
	return filepath.Glob(pattern)
}

func viewProfile(profile string) {
	fmt.Printf("Viewing profile: %s\n", profile)

	var cmd *exec.Cmd
	var url = "http://localhost:8084"

	// Different tools for different profile types
	switch *profileType {
	case "trace":
		// Trace files use `go tool trace`
		cmd = exec.Command("go", "tool", "trace", "-http=:8084", profile)
	default:
		// All other profiles use `go tool pprof`
		cmd = exec.Command("go", "tool", "pprof", "-http=:8084", profile)
	}

	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start viewer: %v", err)
	}

	fmt.Printf("Profile viewer started at %s\n", url)
	if *profileType == "trace" {
		fmt.Println("Note: Trace viewer provides different views than pprof")
	}

	if err := cmd.Wait(); err != nil {
		log.Printf("Viewer process ended: %v", err)
	}
}

func compareProfiles(base, profile string) {
	fmt.Printf("Comparing profiles:\n  Base: %s\n  Current: %s\n", base, profile)

	// Check if profiles are meaningful to compare
	if strings.Contains(base, "goroutine") {
		fmt.Println("Note: Goroutine profile comparison shows changes in goroutine counts and states.")
		fmt.Println("Empty results may indicate similar goroutine patterns between captures.")
	}

	cmd := exec.Command("go", "tool", "pprof", "-http=:8085",
		fmt.Sprintf("-base=%s", base), profile)
	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start comparison: %v", err)
	}

	fmt.Println("Profile comparison started at http://localhost:8085")
	if err := cmd.Wait(); err != nil {
		log.Printf("Comparison viewer process ended: %v", err)
	}
}
