package examples

// StackAllocation demonstrates a variable that stays on the stack
func StackAllocation() int {
	x := 42  // Allocated on the stack
	return x // Return the value (not a reference)
}

// HeapAllocation demonstrates a variable that escapes to the heap
func HeapAllocation() *int {
	x := 42   // Initially on stack
	return &x // Return the address - x escapes to heap
}

// StackWithPointer shows that using pointers doesn't always cause escape
func StackWithPointer() int {
	x := 42
	p := &x  // Address of x used only in this function
	*p = 43  // Modify x through the pointer
	return x // Return the value, not the pointer
}
