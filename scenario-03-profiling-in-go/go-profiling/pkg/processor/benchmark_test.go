package processor

import (
	"fmt"
	"strings"
	"testing"
)

// generateTestDocument creates a test document
func generateTestDocument(size int) *Document {
	var builder strings.Builder
	words := []string{
		"performance", "optimization", "golang", "profiling", "memory",
		"the", "and", "for", "with", "that", "have", "from", "this",
	}

	for i := 0; i < size; i++ {
		builder.WriteString(words[i%len(words)])
		builder.WriteString(" ")
		if i%10 == 0 {
			builder.WriteString(". ")
		}
	}

	return &Document{
		ID:      fmt.Sprintf("test-doc-%d", size),
		Title:   "Test Document",
		Content: builder.String(),
	}
}

func BenchmarkDocumentProcessor(b *testing.B) {
	processor := NewDocumentProcessor()
	doc := generateTestDocument(1_000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = processor.ProcessDocument(doc)
	}
}

func BenchmarkOptimizedProcessor(b *testing.B) {
	processor := NewOptimizedProcessor()
	doc := generateTestDocument(1_000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = processor.ProcessDocument(doc)
	}
}

// Benchmark specific operations
func BenchmarkNormalizeContent(b *testing.B) {
	processor := NewDocumentProcessor()
	content := generateTestDocument(1_000).Content

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = processor.normalizeContent(content)
	}
}

func BenchmarkNormalizeContentOptimized(b *testing.B) {
	processor := NewOptimizedProcessor()
	content := generateTestDocument(1_000).Content

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = processor.normalizeContentOptimized(content)
	}
}
