package examples

// ---------- 1. Passing pointers ----------

// (a) Safe: callee uses the pointer *only* inside its body
func NoEscapeToFunction() int {
	x := 42
	useLocal(&x) // the compiler can see this never saves &x
	return x
}
func useLocal(p *int) { *p += 1 }

// (b) Unsafe: callee stores the pointer in a global
func EscapeToFunction() int {
	x := 42
	savePointer(&x) // &x now leaks outside -> x must escape
	return x
}

var Leaked *int

func savePointer(p *int) { Leaked = p }

// ---------- 2. Large local value ----------

//  1. explicit local: declared with `var x T` and not created by new/make/&{} -> stack limit 128 KiB
//     https://cs.opensource.google/go/go/+/refs/tags/go1.24.2:src/cmd/compile/internal/ir/cfg.go;l=11
//  2. implicit/address taken - created by make/new/&{} or whose address leaves the frame (that includes return by value) -> stack limit 64 KiB
//     https://cs.opensource.google/go/go/+/refs/tags/go1.24.2:src/cmd/compile/internal/ir/cfg.go;l=19
func LargeAllocation() [17_000]int {
	var arr [17_000]int
	for i := range arr {
		arr[i] = i
	}
	return arr
}

// ---------- 3. Slice returned ----------

func SliceEscape() []int {
	s := make([]int, 5) // backing array on heap
	for i := range s {
		s[i] = i
	}
	return s // returning the slice forces escape
}

// ---------- 4. Closure capture ----------

func CapturedByClosureEscape() func() int {
	x := 42
	return func() int { // x lives as long as this closure
		return x
	}
}
