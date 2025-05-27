package processor

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Document represents a text document to be processed
type Document struct {
	ID        string
	Title     string
	Content   string
	WordCount int
	Keywords  []string
	Hash      string
	Processed time.Time
}

// ProcessingResult contains the results of document processing
type ProcessingResult struct {
	DocID       string
	Keywords    []string
	ProcessTime time.Duration
	MemoryUsed  int64
}

// DocumentProcessor processes documents to extract keywords and metadata
type DocumentProcessor struct {
	processedDocs map[string]*Document
	mu            sync.Mutex

	// Intentional issue: Inefficient regex compilation
	stopWords []string
}

// NewDocumentProcessor creates a new document processor
func NewDocumentProcessor() *DocumentProcessor {
	return &DocumentProcessor{
		processedDocs: make(map[string]*Document),
		stopWords: []string{
			"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
			"of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
		},
	}
}

// ProcessDocument processes a single document
func (dp *DocumentProcessor) ProcessDocument(doc *Document) (*ProcessingResult, error) {
	start := time.Now()

	// Intentional issue: Unnecessary string operations
	content := dp.normalizeContent(doc.Content)

	// Intentional issue: Inefficient keyword extraction
	keywords := dp.extractKeywords(content)

	// Intentional issue: Expensive hash calculation
	doc.Hash = dp.calculateHash(doc.Content)

	// Store processed document (memory leak)
	dp.mu.Lock()
	dp.processedDocs[doc.ID] = doc
	dp.mu.Unlock()

	doc.Keywords = keywords
	doc.Processed = time.Now()

	return &ProcessingResult{
		DocID:       doc.ID,
		Keywords:    keywords,
		ProcessTime: time.Since(start),
	}, nil
}

// normalizeContent normalizes the document content
func (dp *DocumentProcessor) normalizeContent(content string) string {
	// Intentional issue: Multiple passes over the same string
	content = strings.ToLower(content)
	content = strings.ReplaceAll(content, "\n", " ")
	content = strings.ReplaceAll(content, "\t", " ")
	content = strings.ReplaceAll(content, ".", " ")
	content = strings.ReplaceAll(content, ",", " ")
	content = strings.ReplaceAll(content, "!", " ")
	content = strings.ReplaceAll(content, "?", " ")
	content = strings.ReplaceAll(content, ";", " ")
	content = strings.ReplaceAll(content, ":", " ")

	// Intentional issue: Regex compiled every time
	re := regexp.MustCompile(`\s+`)
	content = re.ReplaceAllString(content, " ")

	return strings.TrimSpace(content)
}

// extractKeywords extracts keywords from content
func (dp *DocumentProcessor) extractKeywords(content string) []string {
	words := strings.Fields(content)
	wordFreq := make(map[string]int)

	// Intentional issue: Inefficient stop word checking
	for _, word := range words {
		isStopWord := false
		for _, stopWord := range dp.stopWords {
			if word == stopWord {
				isStopWord = true
				break
			}
		}

		if !isStopWord && len(word) > 3 {
			wordFreq[word]++
		}
	}

	// Intentional issue: Inefficient sorting
	var keywords []string
	for word, freq := range wordFreq {
		if freq > 2 {
			// Creating unnecessary string allocations
			keywords = append(keywords, fmt.Sprintf("%s:%d", word, freq))
		}
	}

	return keywords
}

// calculateHash calculates document hash
func (dp *DocumentProcessor) calculateHash(content string) string {
	// Intentional issue: Using MD5 repeatedly on large content
	hasher := md5.New()

	// Intentional issue: Inefficient byte conversion
	for i := 0; i < 10; i++ {
		hasher.Write([]byte(content))
	}

	return hex.EncodeToString(hasher.Sum(nil))
}

// GetProcessedCount returns the number of processed documents
func (dp *DocumentProcessor) GetProcessedCount() int {
	dp.mu.Lock()
	defer dp.mu.Unlock()
	return len(dp.processedDocs)
}
