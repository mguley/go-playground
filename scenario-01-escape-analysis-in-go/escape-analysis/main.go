package main

import (
	"escapeanalysis/examples"
	"fmt"
	"runtime"
)

func main() {

	fmt.Println("Go Escape Analysis Examples")
	fmt.Printf("Go version: %s\n", runtime.Version())

	fmt.Println("\n--- Common Escape Scenarios ---")

	fmt.Printf("No-escape pointer result: %d\n", examples.NoEscapeToFunction())
	fmt.Printf("Escape-to-function result: %d (leaked = %d)\n",
		examples.EscapeToFunction(), *examples.Leaked) // expose the leak

	large := examples.LargeAllocation()
	fmt.Printf("Large allocation first element: %d\n", large[0])

	slice := examples.SliceEscape()
	fmt.Printf("Slice escape length: %d\n", len(slice))

	closure := examples.CapturedByClosureEscape()
	fmt.Printf("Closure result: %d\n", closure())
}
