package examples

import (
	"runtime"
	"testing"
)

// A payload big enough to notice copy vs. indirection
type Point struct {
	X, Y, Z int64
	Name    string
	Data    [100]byte
}

var (
	sinkInt   int64  // prevents dead-code elimination for int results
	sinkPoint *Point // holds a pointer so it must live past the call
)

// ---------- 1. Pure-stack variant ----------

//go:noinline
func makeStackPoint(i int) Point {
	return Point{
		X: int64(i), Y: int64(i), Z: int64(i),
		Name: "point",
	}
}

func BenchmarkPointStack(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := makeStackPoint(i)
		sinkInt = p.X
	}
}

// ---------- 2. Heap variant (returns *Point) ----------

//go:noinline
func makeHeapPoint(i int) *Point {
	// Returning &local forces it to escape once inlining is disabled.
	return &Point{
		X: int64(i), Y: int64(i), Z: int64(i),
		Name: "point",
	}
}

func BenchmarkPointHeap(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkPoint = makeHeapPoint(i) // pointer survives the call → heap
		sinkInt = sinkPoint.X + 1
	}
}

// ---------- 3. Slice variants ----------

func BenchmarkSliceStack(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var s [16]int // lives on the stack
		runtime.KeepAlive(s)
	}
}

var sliceSink []int

func BenchmarkSliceHeap(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sliceSink = make([]int, 16) // always heap
	}
}
