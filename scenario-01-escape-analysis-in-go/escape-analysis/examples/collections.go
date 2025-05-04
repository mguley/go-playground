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
