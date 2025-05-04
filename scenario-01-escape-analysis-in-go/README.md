# Escape Analysis in Go: Understanding Memory Optimization

## Table of Contents
- [Introduction](#introduction)
- [What is Escape Analysis?](#what-is-escape-analysis)
- [Prerequisites](#prerequisites)
- [Step 1: Setting Up Our Environment](#step-1-setting-up-our-environment)
- [Step 2: Basic Escape Analysis Examples](#step-2-basic-escape-analysis-examples)
- [Step 3: Common Escape Scenarios](#step-3-common-escape-scenarios)
- [Step 4: Reading the Compiler's Escape Analysis Output](#step-4-reading-the-compilers-escape-analysis-output)
- [Step 5: Function Returns and Interfaces](#step-5-function-returns-and-interfaces)
- [Step 6: Slices, Maps, and Escape Analysis](#step-6-slices-maps-and-escape-analysis)
- [Step 7: Benchmarking Escape vs. Non-Escape](#step-7-benchmarking-escape-vs-non-escape)
- [Understanding What Happened Under the Hood](#understanding-what-happened-under-the-hood)
- [Conclusion](#conclusion)

#### Introduction

In high-performance Go applications, memory management isn't just about preventing leaks - it's about optimizing where your data lives.

For developers building systems that handle thousands of requests per second, understanding how variables are allocated 
can significantly impact performance:

- A financial trading platform where nanoseconds matter might see orders execute 30% faster
- An API service could handle twice the load on the same hardware
- A data processing pipeline might reduce GC pauses from seconds to milliseconds

While Go abstracts away most memory management concerns, truly performant applications require developers to understand 
what happens beneath the abstraction.

In this deep dive, we'll explore how escape analysis works, how to detect when variables `"escape"` to the heap,
and practical techniques to keep allocations on the stack when it matters most.

#### What is Escape Analysis?

Escape analysis is a compile-time process that determines where a variable should be allocated: on the `stack` or on the `heap`.

In Go, memory is divided into two primary regions:

1. **Stack**:
    - Each goroutine starts with a small stack (about 2KiB) that grows and shrinks on demand
    - Creating or using stack variables is almost free (usually just a pointer adjustment)
    - Memory is reclaimed automatically when functions return
    - The garbage collector does not manage stack memory, but it does scan live stack frames for pointers during every GC cycle

2. **Heap**:
    - A global region managed by Go's concurrent mark-and-sweep garbage collector
    - Data placed here can outlive the function that created it, so the GC must periodically mark reachable objects, sweep unreachable ones, and return that memory
      to the allocator
    - Those GC activities add CPU time proportional to the amount of live heap and the number of pointers that must be scanned

The compiler's escape analysis tries to allocate as much as possible on the stack for performance, but certain situations 
force variables to `"escape"` to the heap:

- When a variable's lifetime can't be determined at compile time
- When a variable is too large for the stack
- When a variable's address is shared with other functions or goroutines

Understanding these mechanisms allows us to write code that minimizes heap allocations and reduces garbage collection pressure,
which is critical for high-performance Go applications.

#### Prerequisites

Before we begin, you'll need:
- Go installed (version 1.24+)
- Basic knowledge of Go syntax and concepts
- A code editor of your choice
- Terminal/command-line access

#### Step 1: Setting Up Our Environment

Let's create a directory structure for our escape analysis experiments:

```bash
mkdir -p escape-analysis/examples
cd escape-analysis
go mod init escapeanalysis
touch main.go
```

Now, let's set up our `main.go` file with a basic structure:

```go
package main

import (
	"fmt"
	"runtime"
)

func main() {
	
	fmt.Println("Go Escape Analysis Examples")
	fmt.Printf("Go version: %s\n", runtime.Version())

	// We'll add our examples here
}
```

To run this with escape analysis output enabled:
```bash
go build -gcflags="-m -l" main.go
```

The `-gcflags="-m -l"` flag tells the compiler to print escape analysis decisions. The `-l` disables inlining, which makes the output easier to understand.

#### Step 2: Basic Escape Analysis Examples

Let's create our first example to demonstrate the difference between stack and heap allocations.

Create a file `examples/basic.go` with these functions:

```go
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
```

Compile the `package` only to see escape analysis results:
```bash
go build -gcflags="-m -l" ./examples
```

Expected output:
```
examples/basic.go:11:2: moved to heap: x
```

Note that there are no messages for `StackAllocation` or `StackWithPointer` because these variables stay on the stack as intended.

`Why isolate the package?`

Passing the results through the `fmt.Printf` (as `main.go` does) forces additional heap boxes - because `fmt` stores every
argument in an `interface{}` slot that resides on the heap.

This does not change where `x` was first allocated; it merely reflects the cost of printing.

Now modify `main.go` to use these functions:

```go
package main

import (
	"escapeanalysis/examples"
	"fmt"
	"runtime"
)

func main() {

	fmt.Println("Go Escape Analysis Examples")
	fmt.Printf("Go version: %s\n", runtime.Version())

	// Basic examples
	val1 := examples.StackAllocation()
	val2 := examples.HeapAllocation()
	val3 := examples.StackWithPointer()

	fmt.Printf("Stack allocation result: %d\n", val1)
	fmt.Printf("Heap allocation result: %d\n", *val2)
	fmt.Printf("Stack with pointer result: %d\n", val3)
}
```

Let's analyze what the compiler tells us:
```bash
go build -gcflags="-m -l" ./main.go
```

And you should see the following output:
```
./main.go:11:13: ... argument does not escape
./main.go:11:14: "Go Escape Analysis Examples" escapes to heap
./main.go:12:12: ... argument does not escape
./main.go:12:48: runtime.Version() escapes to heap
./main.go:19:12: ... argument does not escape
./main.go:19:46: val1 escapes to heap
./main.go:20:12: ... argument does not escape
./main.go:20:45: *val2 escapes to heap
./main.go:21:12: ... argument does not escape
./main.go:21:48: val3 escapes to heap
```

These refer to the temporary interface boxes created by `fmt.Printf`, not to the original variables inside `StackAllocation`,
`HeapAllocation`, `StackWithPointer`.

When you need a precise view of a library's allocations, compile the library by itself, not the program that uses it.
This helps avoid confusion from temporary allocations created by functions like `fmt.Printf`.

#### Step 3: Common Escape Scenarios

Most heap allocations are triggered by four common patterns:
1. **Passing a pointer to another function**: The callee might store the pointer for later.
2. **Very large local value**: Protects the stack from megabyte bursts.
3. **Returning a slice/map**: The backing array or map header must outlive the function.
4. **Capturing a variable in a closure**: The closure can run after its creator returns.

We will reproduce each pattern and compare it with a safe variant that stays on the stack.

Let's create `examples/scenarios.go` to demonstrate these patterns:

```go
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

// 1. explicit local: declared with `var x T` and not created by new/make/&{} -> stack limit 128 KiB
//    https://cs.opensource.google/go/go/+/refs/tags/go1.24.2:src/cmd/compile/internal/ir/cfg.go;l=11
// 2. implicit/address taken - created by make/new/&{} or whose address leaves the frame (that includes return by value) -> stack limit 64 KiB
//    https://cs.opensource.google/go/go/+/refs/tags/go1.24.2:src/cmd/compile/internal/ir/cfg.go;l=19  
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
```

Compile only the package to see escape analysis results:
```bash
go build -gcflags="-m -l" ./examples/scenarios.go
```

Expected output:
```
examples/scenarios.go:11:15: p does not escape                       # 1a - safe pointer

examples/scenarios.go:22:18: leaking param: p                        # 1b - unsafe pointer
examples/scenarios.go:15:2: moved to heap: x                         # 1b - unsafe pointer

examples/scenarios.go:31:6: moved to heap: arr                       # 2 - large value, 128 KiB | 64 KiB

examples/scenarios.go:41:11: make([]int, 5) escapes to heap          # 3 - slice returned

examples/scenarios.go:52:9: func literal escapes to heap             # 4 - closure capture
```

`Why the -l flag again?`

Disabling inlining guarantees that `useLocal` / `savePointer` remain separate call sites; that keeps the demo predictable.

`What each message means`

| Message fragment                 | Explanation                                                       |
|---------------------------------|-------------------------------------------------------------------|
| p does not escape                | Safe pointer: the pointer `p` does not escape the function call.  |
| leaking param: p                | Unsafe pointer: the parameter `p` leaks outside the function.     |
| moved to heap: x                | Variable `x` was moved to the heap due to escape from scope.      |
| moved to heap: arr              | Large local value `arr` moved to heap (exceeds stack size limit). |
| make([]int, 5) escapes to heap | Slice backing array must live on the heap because it escapes.     |
| func literal escapes to heap    | Closure object and captured variables escape to the heap.         |

Add this block to `main.go` to test these scenarios:

```go
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
      examples.EscapeToFunction(), *examples.Leaked)   // expose the leak

   large := examples.LargeAllocation()
   fmt.Printf("Large allocation first element: %d\n", large[0])

   slice := examples.SliceEscape()
   fmt.Printf("Slice escape length: %d\n", len(slice))

   closure := examples.CapturedByClosureEscape()
   fmt.Printf("Closure result: %d\n", closure())
}
```

Re-build the whole program if you like, but remember that the extra messages you will see now are mostly the boxing overhead
from those `fmt.Printf` calls.
```bash
go build -gcflags="-m -l" ./main.go
```

Quick reference cheat sheet:
1. **Passing pointers to functions**: When you pass a pointer to a function, the compiler can't always prove the pointer won't be stored somewhere that outlives the function call.
2. **Large allocations**: As the stack is limited in size, very large values might be placed on the heap regardless of escaping rules.
3. **Slices and maps**: These data structures contain internal pointers and often escape to the heap, especially when returned from functions.
4. **Closures**: Variables captured by closures must escape to the heap since the closure could be called after the original function returns.

#### Step 4: Reading the Compiler's Escape Analysis Output

The Go compiler's escape analysis output can be cryptic at first. Let's create a file to practice reading these messages `examples/reading_output.go`:

```go
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
```

Compile with detailed escape analysis:
```bash
go build -gcflags="-m=2 -l" ./examples/reading_output.go
```

The `-m=2` flag provides more verbose escape analysis output. You can pass `-m=2` (or repeat `-m`) to see the detailed flow
lines that explain why something escaped.

Start with plain `-m` for a terse list, and add more `m's` only when you need the lineage.

Output:
```
examples/reading_output.go:18:17: p does not escape

examples/reading_output.go:21:21: parameter p leaks to {heap} with derefs=0:
examples/reading_output.go:21:21:   flow: {heap} = p:
examples/reading_output.go:21:21:     from global = p (assign) at examples/reading_output.go:21:38
examples/reading_output.go:21:21: leaking param: p

examples/reading_output.go:23:23: a does not escape

examples/reading_output.go:8:2: z escapes to heap:
examples/reading_output.go:8:2:   flow: {heap} = &z:
examples/reading_output.go:8:2:     from &z (address-of) at examples/reading_output.go:13:17
examples/reading_output.go:8:2:     from pointerEscaper(&z) (call parameter) at examples/reading_output.go:13:16
examples/reading_output.go:8:2: moved to heap: z

examples/reading_output.go:14:19: i does not escape
```

Decoding the messages:

| Message fragment                                       | Meaning                                                                                                                                                                                                                  |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`p does not escape`**                                | The pointer argument is used only inside `usePointer`; it stays on the caller’s stack.                                                                                                                                   |
| **`parameter p leaks to {heap}` / `leaking param: p`** | The compiler proved that **the *parameter itself* must reside on the heap**, because the callee stores it somewhere (here: the global var).                                                                              |
| **`a does not escape`**                                | A plain local variable; kept on the stack.                                                                                                                                                                               |
| **`z escapes to heap:` … `moved to heap: z`**          | The address `&z` flows into `pointerEscaper`, so `z` must live on the heap. The two lines appear together: `escapes to heap` shows the flow, `moved to heap` confirms the actual relocation.                             |
| **`i does not escape`**                                | In Go 1.24 the simple assignment of a small integer to an `interface{}` *inside the same function* no longer forces an escape. The compiler places the pair (type + value) in a pre‑allocated scratch slot on the stack. |

This verbose output helps us understand exactly which variables are escaping and why.

#### Step 5: Function Returns and Interfaces

Let's explore how function returns and interface conversions affect escape analysis in `examples/returns_interfaces.go`:

```go
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
```

Compile with detailed escape analysis:
```bash
go build -gcflags="-m=2 -l" ./examples/returns_interfaces.go
```

Output:
```
examples/returns_interfaces.go:13:2: s escapes to heap:
examples/returns_interfaces.go:13:2:   flow: ~r0 = &s:
examples/returns_interfaces.go:13:2:     from &s (address-of) at examples/returns_interfaces.go:14:9
examples/returns_interfaces.go:13:2:     from return &s (return) at examples/returns_interfaces.go:14:2
examples/returns_interfaces.go:13:2: moved to heap: s

examples/returns_interfaces.go:21:6: x does not escape

examples/returns_interfaces.go:29:6: x escapes to heap:
examples/returns_interfaces.go:29:6:   flow: i = &{storage for x}:
examples/returns_interfaces.go:29:6:     from x (spill) at examples/returns_interfaces.go:29:6
examples/returns_interfaces.go:29:6:     from i = x (assign) at examples/returns_interfaces.go:29:4
examples/returns_interfaces.go:29:6:   flow: {storage for ... argument} = i:
examples/returns_interfaces.go:29:6:     from ... argument (slice-literal-element) at examples/returns_interfaces.go:30:13
examples/returns_interfaces.go:29:6:   flow: {heap} = {storage for ... argument}:
examples/returns_interfaces.go:29:6:     from ... argument (spill) at examples/returns_interfaces.go:30:13
examples/returns_interfaces.go:29:6:     from fmt.Println(... argument...) (call parameter) at examples/returns_interfaces.go:30:13
examples/returns_interfaces.go:29:6: x escapes to heap

examples/returns_interfaces.go:30:13: ... argument does not escape

examples/returns_interfaces.go:45:9: 42 escapes to heap:
examples/returns_interfaces.go:45:9:   flow: ~r0 = &{storage for 42}:
examples/returns_interfaces.go:45:9:     from 42 (spill) at examples/returns_interfaces.go:45:9
examples/returns_interfaces.go:45:9:     from return 42 (return) at examples/returns_interfaces.go:45:2

examples/returns_interfaces.go:43:10: s escapes to heap:
examples/returns_interfaces.go:43:10:   flow: ~r0 = &{storage for s}:
examples/returns_interfaces.go:43:10:     from s (spill) at examples/returns_interfaces.go:43:10
examples/returns_interfaces.go:43:10:     from return s (return) at examples/returns_interfaces.go:43:3
examples/returns_interfaces.go:43:10: s escapes to heap

examples/returns_interfaces.go:45:9: 42 escapes to heap
```

| Pattern                                                       | Does it escape?             | Why                                                                                                                            |
|---------------------------------------------------------------| --------------------------- |--------------------------------------------------------------------------------------------------------------------------------|
| **Return by value (`ReturnValue`)**                           | **No**                      | The string header is copied to the caller’s stack; the literal bytes live in read‑only data.                                   |
| **Return pointer (`ReturnPointer`)**                          | **Yes** (`s` moved to heap) | The pointed‑to variable must outlive the function.                                                                             |
| **Interface ⟶ used locally (`InterfaceOnStack`)**             | **No**                      | Since the interface value never leaves the frame, the compiler places the `(type, value)` pair in a scratch slot on the stack. |
| **Interface ⟶ passed on (`InterfaceEscapes`)**                | **Yes**                     | Passing the interface to `fmt.Println` transfers ownership; the compiler can no longer prove it dies in‑frame, so the value is boxed on the heap. |
| **Pure arithmetic (`NoEscape`)**                              | **No**                      | Plain scalars that never have their address taken remain on the stack or in registers.                                         |
| **Conditional return of `interface{}` (`ConditionalEscape`)** | **Yes (both branches)**     | The compiler must choose a single storage strategy for `~r0`; because *one* branch clearly escapes, the conservative choice is “heap for all”. |

Important patterns to notice:
1. **Returning a value is cheap**: Go copies the data (or just a header) and keeps it on the stack whenever possible.
2. **Any returned pointer guarantees a heap move** for the pointed-to variable.
3. **Interface conversions are nuanced**:
   - It stays on the stack if the compiler can prove the interface dies in the same frame
   - It escapes the moment it is returned, stored globally, or passed to another function

#### Step 6: Slices, Maps, and Escape Analysis

Collections like slices and maps have special behavior with regard to escape analysis. Let's create `examples/collections.go`:

```go
package examples

// ---------- 1. Slices ----------

// 1-a small slice, used locally ➜ stays on the stack
func SliceLocal() int {
	s := make([]int, 3)
	s[0], s[1], s[2] = 1, 2, 3
	return s[0] + s[1] + s[2]
}

// 1-b same size but *returned* ➜ backing array must escape
func SliceReturn() []int {
	s := make([]int, 3)
	return s
}

// 1-c large slice ➜ size limit forces heap
func SliceBig() {
	_ = make([]int, 20_000) // stack limit 64 KiB via make, will escape to heap
}

// ---------- 2. Maps ----------

// 2-a map that never grows past its first bucket ➜ entire hmap + bucket live on the stack
func SmallMapSum() int {
	m := make(map[int]int)
	for i := 0; i < 5; i++ {
		m[i] = i
	}
	sum := 0
	for _, v := range m {
		sum += v
	}
	return sum // m dies here
}

// 2-b map that grows or is returned ➜ escapes
func MapEscapes() map[int]int {
	m := make(map[int]int, 20)
	return m // escape
}

// ---------- 3. Array returned by value ----------

func SmallArray() [3]int {
	return [3]int{1, 2, 3}
}
```

Compile with detailed escape analysis:
```bash
go build -gcflags="-m=2 -l" ./examples/collections.go
```

Output:
```
# 1-a small slice, local
examples/collections.go:7:11: make([]int, 3) does not escape

# 1-b same slice, returned
examples/collections.go:14:11: make([]int, 3) escapes to heap:
examples/collections.go:14:11:   flow: s = &{storage for make([]int, 3)}:
examples/collections.go:14:11:     from make([]int, 3) (spill) at examples/collections.go:14:11
examples/collections.go:14:11:     from s := make([]int, 3) (assign) at examples/collections.go:14:4
examples/collections.go:14:11:   flow: ~r0 = s:
examples/collections.go:14:11:     from return s (return) at examples/collections.go:15:2
examples/collections.go:14:11: make([]int, 3) escapes to heap

# 1-c big slice
examples/collections.go:20:10: make([]int, 20000) escapes to heap:
examples/collections.go:20:10:   flow: {heap} = &{storage for make([]int, 20000)}:
examples/collections.go:20:10:     from make([]int, 20000) (too large for stack) at examples/collections.go:20:10
examples/collections.go:20:10: make([]int, 20000) escapes to heap

# 2-a small map
examples/collections.go:27:11: make(map[int]int) does not escape

# 2-b map, returned
examples/collections.go:40:11: make(map[int]int, 20) escapes to heap:
examples/collections.go:40:11:   flow: m = &{storage for make(map[int]int, 20)}:
examples/collections.go:40:11:     from make(map[int]int, 20) (spill) at examples/collections.go:40:11
examples/collections.go:40:11:     from m := make(map[int]int, 20) (assign) at examples/collections.go:40:4
examples/collections.go:40:11:   flow: ~r0 = m:
examples/collections.go:40:11:     from return m (return) at examples/collections.go:41:2
examples/collections.go:40:11: make(map[int]int, 20) escapes to heap
```

Key insights about collections:
1. **Slices have underlying arrays** that often escape to the heap
2. **Small arrays returned by value** might not escape, as they're copied
3. **Size matters**: the larger the collection, the more likely it escapes.

#### Step 7: Benchmarking Escape vs. Non-Escape

Benchmarks are an excellent way to see the cost of heap allocation. We'll measure three things:
- **ns/op**: CPU time per operation
- **B/op**: Bytes allocated per operation
- **allocs/op**: Number of distinct heap objects per operation

Let's create a benchmark to measure the performance impact of stack vs. heap allocations in `examples/benchmark_test.go`:

```go
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
```

Run the benchmarks:
```bash
go test -C examples -bench=. -benchmem -gcflags=all=-l
```

| Flag                  | Why we need it                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`-C examples`**     | Runs the tests inside the `examples` module without changing your current directory.                                                                                                                                                                                                                                                                                                                                                          |
| **`-bench=.`**        | Execute every function whose name starts with `Benchmark…`. (`.` is a regexp that matches everything.)                                                                                                                                                                                                                                                                                                                                        |
| **`-benchmem`**       | In addition to timing, print *bytes allocated* and *number of allocations* per operation.                                                                                                                                                                                                                                                                                                                                                     |
| **`-gcflags=all=-l`** | “Tell the compiler **not to inline any function** in *any* package.”<br>Inlining is great for production code, but in micro‑benchmarks it can accidentally hide heap allocations (the compiler proves the pointer never escapes once the helper is inlined).  Disabling inlining guarantees that the helper functions (`makeStackPoint`, `makeHeapPoint`, …) are kept separate, so their escape decisions are made in their own stack frames. |

Output:
```
goos: linux
goarch: amd64

BenchmarkPointStack-24    	130878974	          9.084 ns/op	       0 B/op	       0 allocs/op
BenchmarkPointHeap-24     	21043689	          55.75 ns/op	     144 B/op	       1 allocs/op
BenchmarkSliceStack-24    	645727572	          1.958 ns/op	       0 B/op	       0 allocs/op
BenchmarkSliceHeap-24     	28888604	          40.01 ns/op	     128 B/op	       1 allocs/op
PASS
ok  	escapeanalysis/examples	7.758s
```

Reading the numbers:

| Column        | What it means                                                              | What we see                                                                                                 |
| ------------- | -------------------------------------------------------------------------- |-------------------------------------------------------------------------------------------------------------|
| **ns/op**     | Average *CPU time* per call.                                               | Heap versions are **slower** because of extra pointer chasing, more instructions, and GC bookkeeping. |
| **B/op**      | Actual *bytes* handed to the allocator (includes object header & padding). | `Point` costs 144 bytes; the 16‑int slice costs 128 bytes.  Stack variants stay at **0 B/op**.              |
| **allocs/op** | How many distinct heap objects were created.                               | Each “heap” function allocates **exactly one object**; the stack versions allocate **none**.                |

`Why do the stack versions still consume a few nanoseconds?`

Creating a stack variable is just pointer arithmetic, but it still executes load/store instructions and runs inside a tight loop,
so some CPU time is expected.

Checklist for your code:

| Checklist Item       | What to look for                                                                                 |
|----------------------|--------------------------------------------------------------------------------------------------|
| `allocs/op == 0`     | Fast path: the function did **not** allocate.                                                    |
| `allocs/op > 0`      | Something escaped. Use `go build -gcflags="-m=2"` on that package to find out why.               |
| `B/op`               | Large objects on the heap hurt twice: they take longer to allocate **and** they enlarge GC work. |
| `ns/op` spike        | Suspicious if heap and stack variants perform the same task but differ wildly in time.           |

Once you understand how to read these three columns, you can quickly spot allocation hot spots in any code base, even before you reach for the full profiler.

Run the benchmark with `-cpu 1,2,4` for more realistic throughput numbers to see how allocations scale on multiple cores.
```bash
go test -C examples -bench=. -benchmem -gcflags=all=-l -cpu 1,2,4
```

Here are key optimization techniques for reducing heap allocations:
1. **Pre-size collections** when the size is known or can be estimated
2. **Reuse buffers** instead of creating new ones
3. **Accept buffers from callers** to let them manage memory
4. **Use structs with methods** instead of loose functions to avoid interface conversions
5. **Prefer arrays to slices** for small, fixed-size collections
6. **Use sync.Pool** for frequently allocated objects that can be reused

#### Understanding What Happened Under the Hood

Now that we've explored escape analysis from multiple angles, let's understand what's happening at a deeper level:
1. **Compile-Time Analysis**: Escape analysis happens during compilation, not at runtime.
2. **Conservative Approach**: If there's any doubt about whether a variable needs to escape, the compiler chooses the safer path - heap allocation.
This ensures program correctness at the cost of performance.
3. **Implementation Details Matter**: The Go escape analysis implementation can change between versions. 
This is why it's important to benchmark and profile rather than rely solely on intuition.
4. **Looking at Assembly**: For the curious, examining the generated assembly code (via `go build -gcflags="-S"`) shows exactly 
how variables are allocated and accessed. Stack variables use offsets from the stack pointer, while heap variables require additional indirection.
5. **Compiler Optimizations**: Besides escape analysis, the Go compiler performs many other optimizations, like inlining small functions, eliminating dead code, and constant propagation.
These can affect escape analysis results in non-obvious ways.

The beauty of Go's approach is that it handles this complexity automatically while giving developers tools to understand and optimize when needed.
You don't need to manually manage memory as in C/C++, but you have visibility into memory behavior when performance matters.

#### Conclusion

Escape analysis sits at the intersection of Go's high-level programming model and its performance-oriented design. 
By understanding this mechanism, you can write code that's both idiomatic and efficient.

Key takeaways:
- Stack allocations are significantly faster than heap allocations and put no pressure on the garbage collector
- Returning pointers to local variables is safe in Go but causes heap allocations
- Interfaces, closures, and function pointers frequently cause escapes
- Collection types often end up on the heap due to their internal pointers
- Optimization techniques like buffer reuse can dramatically reduce allocations

Rather than prematurely optimizing all code, use these techniques selectively on hot paths identified through profiling. 
Most applications will benefit most from focusing optimizations on the 10% of code that consumes 90% of resources.