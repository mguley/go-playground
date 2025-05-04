package examples

var global *int

func ReadingOutput() {
	x := 42  // Case 1 - local, no escape
	y := 100 // Case 2 - pointer arg, no escape
	z := 200 // Case 3 - escapes
	i := 300 // Case 4 - interface conversion

	x = x + 1
	usePointer(&y)
	pointerEscaper(&z)
	interfaceEscaper(i)
}

// Function that takes a pointer but doesn't store it anywhere
func usePointer(p *int) { *p++ } // safe

// Function that saves the pointer, forcing escape
func pointerEscaper(p *int) { global = p } // leaks

func interfaceEscaper(a any) { _ = a }
