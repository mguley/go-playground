package examples

import "fmt"

// --- 1. Returning by value: stays on stack
func ReturnValue() string {
	s := "hello"
	return s // header copied – no escape
}

// --- 2. Returning a pointer: forces escape
func ReturnPointer() *string {
	s := "hello"
	return &s // &s lives longer than the frame, will escape to heap
}

// --- 3a. Interface conversion used *locally* – stays on stack
func InterfaceOnStack() {
	x := 42
	var i any
	i = x // compiler keeps the pair on the stack
	_ = i // no call, no return ➜ safe
}

// --- 3b. Interface passed to another function – escapes
func InterfaceEscapes() {
	x := 42
	var i any
	i = x          // same as above…
	fmt.Println(i) // …but now it is handed off, so it escapes
}

// --- 4. No escape at all
func NoEscape() int {
	a, b := 10, 20
	return a + b
}

// --- 5. Flow-dependent return of interface{}
func ConditionalEscape(flag bool) any {
	if flag {
		s := "escaped"
		return s // s must live past the frame, will escape to heap
	}
	return 42 // constants need storage too, will escape to heap
}
