package processor

import (
	"crypto/md5"
	"encoding/hex"
	"regexp"
	"strings"
	"sync"
	"time"
)

var (
	// Pre-compile regex at the package level
	whitespaceRegex = regexp.MustCompile(`\s+`)

	// Use map for O(1) stop word lookup
	stopWordsMap = map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "of": true, "with": true, "by": true, "from": true,
		"as": true, "is": true, "was": true, "are": true, "were": true,
		"been": true,
	}
)

// OptimizedProcessor is an optimized version of DocumentProcessor
type OptimizedProcessor struct {
	cache *LRUCache

	// Pre-allocated buffers
	bufferPool sync.Pool
}

// NewOptimizedProcessor creates an optimized processor
func NewOptimizedProcessor() *OptimizedProcessor {
	return &OptimizedProcessor{
		cache: NewLRUCache(1_000, 5*time.Minute), // Max 1_000 items, 5 min TTL
		bufferPool: sync.Pool{
			New: func() interface{} {
				return new(strings.Builder)
			},
		},
	}
}

// ProcessDocument processes a document efficiently
func (op *OptimizedProcessor) ProcessDocument(doc *Document) (*ProcessingResult, error) {
	start := time.Now()

	// Use optimized normalization
	content := op.normalizeContentOptimized(doc.Content)

	// Use optimized keyword extraction
	keywords := op.extractKeywordsOptimized(content)

	// Use optimized hash calculation
	doc.Hash = op.calculateHashOptimized(doc.Content)

	// Store in a bounded cache instead of an unbounded map
	op.cache.Put(doc.ID, doc)

	doc.Keywords = keywords
	doc.Processed = time.Now()

	return &ProcessingResult{
		DocID:       doc.ID,
		Keywords:    keywords,
		ProcessTime: time.Since(start),
	}, nil
}

// normalizeContentOptimized efficiently normalizes content
func (op *OptimizedProcessor) normalizeContentOptimized(content string) string {
	// Get a string builder from the pool
	builder := op.bufferPool.Get().(*strings.Builder)
	defer func() {
		builder.Reset()
		op.bufferPool.Put(builder)
	}()

	// Single pass normalization
	builder.Grow(len(content)) // Pre-allocate capacity

	for i, r := range content {
		switch r {
		case '\n', '\t', '.', ',', '!', '?', ';', ':':
			builder.WriteByte(' ')
		default:
			// Convert to lowercase inline
			if r >= 'A' && r <= 'Z' {
				builder.WriteByte(byte(r + 32))
			} else {
				builder.WriteRune(r)
			}
		}
		_ = i // Use index to avoid unused variable warning
	}

	// Use pre-compiled regex
	normalized := whitespaceRegex.ReplaceAllString(builder.String(), " ")
	return strings.TrimSpace(normalized)
}

// extractKeywordsOptimized efficiently extracts keywords
func (op *OptimizedProcessor) extractKeywordsOptimized(content string) []string {
	words := strings.Fields(content)
	wordFreq := make(map[string]int, len(words)/10) // Pre-size map

	// Single pass with map lookup
	for _, word := range words {
		// O(1) stop word check
		if !stopWordsMap[word] && len(word) > 3 {
			wordFreq[word]++
		}
	}

	// Pre-allocate result slice
	keywords := make([]string, 0, len(wordFreq))
	for word, freq := range wordFreq {
		if freq > 2 {
			keywords = append(keywords, word)
		}
	}

	return keywords
}

// calculateHashOptimized efficiently calculates hash
func (op *OptimizedProcessor) calculateHashOptimized(content string) string {
	// Single hash calculation
	hasher := md5.New()
	hasher.Write([]byte(content))
	return hex.EncodeToString(hasher.Sum(nil))
}
