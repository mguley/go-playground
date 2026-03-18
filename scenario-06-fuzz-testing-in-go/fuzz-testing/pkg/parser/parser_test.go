package parser

import (
	"testing"
)

// TestParseBasicKeyValue tests simple key-value parsing.
func TestParseBasicKeyValue(t *testing.T) {
	input := `name = John`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if doc.Len() != 1 {
		t.Fatalf("expected 1 entry, got %d", doc.Len())
	}

	entry, ok := doc.Get("name")
	if !ok {
		t.Fatal("key 'name' not found")
	}
	if entry.Value != "John" {
		t.Errorf("expected value 'John', got %q", entry.Value)
	}
	if entry.Type != TypeString {
		t.Errorf("expected type string, got %s", entry.Type)
	}
}

// TestParseQuotedString tests quoted string values with spaces.
func TestParseQuotedString(t *testing.T) {
	input := `greeting = "hello world"`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	entry, ok := doc.Get("greeting")
	if !ok {
		t.Fatal("key 'greeting' not found")
	}
	if entry.Value != "hello world" {
		t.Errorf("expected 'hello world', got %q", entry.Value)
	}
	if entry.Type != TypeString {
		t.Errorf("expected type string, got %s", entry.Type)
	}
}

// TestParseEscapeSequences tests escape handling in quoted strings.
func TestParseEscapeSequences(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "escaped quote",
			input:    `msg = "she said \"hello\""`,
			expected: `she said "hello"`,
		},
		{
			name:     "escaped backslash",
			input:    `path = "C:\\Users\\admin"`,
			expected: `C:\Users\admin`,
		},
		{
			name:     "escaped newline",
			input:    `text = "line1\nline2"`,
			expected: "line1\nline2",
		},
		{
			name:     "escaped tab",
			input:    `text = "col1\tcol2"`,
			expected: "col1\tcol2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc, err := Parse(tt.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			entry := doc.Entries[0]
			if entry.Value != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, entry.Value)
			}
		})
	}
}

// TestParseIntegerValues tests integer value parsing.
func TestParseIntegerValues(t *testing.T) {
	input := `
count = 42
negative = -17
zero = 0
`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tests := []struct {
		key   string
		value string
	}{
		{"count", "42"},
		{"negative", "-17"},
		{"zero", "0"},
	}

	for _, tt := range tests {
		entry, ok := doc.Get(tt.key)
		if !ok {
			t.Errorf("key %q not found", tt.key)
			continue
		}
		if entry.Value != tt.value {
			t.Errorf("key %q: expected value %q, got %q", tt.key, tt.value, entry.Value)
		}
		if entry.Type != TypeInteger {
			t.Errorf("key %q: expected type integer, got %s", tt.key, entry.Type)
		}
	}
}

// TestParseFloatValues tests floating-point value parsing.
func TestParseFloatValues(t *testing.T) {
	input := `
pi = 3.14
rate = -0.5
big = 100.0
`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, key := range []string{"pi", "rate", "big"} {
		entry, ok := doc.Get(key)
		if !ok {
			t.Errorf("key %q not found", key)
			continue
		}
		if entry.Type != TypeFloat {
			t.Errorf("key %q: expected type float, got %s", key, entry.Type)
		}
	}
}

// TestParseBooleanValues tests boolean value parsing.
func TestParseBooleanValues(t *testing.T) {
	input := `
enabled = true
debug = false
verbose = True
`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tests := []struct {
		key   string
		value string
	}{
		{"enabled", "true"},
		{"debug", "false"},
		{"verbose", "true"},
	}

	for _, tt := range tests {
		entry, ok := doc.Get(tt.key)
		if !ok {
			t.Errorf("key %q not found", tt.key)
			continue
		}
		if entry.Value != tt.value {
			t.Errorf("key %q: expected %q, got %q", tt.key, tt.value, entry.Value)
		}
		if entry.Type != TypeBoolean {
			t.Errorf("key %q: expected type boolean, got %s", tt.key, entry.Type)
		}
	}
}

// TestParseComments tests that comments and empty lines are ignored.
func TestParseComments(t *testing.T) {
	input := `
# This is a comment
name = Alice

# Another comment
age = 30
`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if doc.Len() != 2 {
		t.Errorf("expected 2 entries, got %d", doc.Len())
	}
}

// TestParseDottedKeys tests hierarchical key names.
func TestParseDottedKeys(t *testing.T) {
	input := `
database.host = "localhost"
database.port = 5432
database.name = "myapp"
`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	entry, ok := doc.Get("database.host")
	if !ok {
		t.Fatal("key 'database.host' not found")
	}
	if entry.Value != "localhost" {
		t.Errorf("expected 'localhost', got %q", entry.Value)
	}
}

// TestParseEmptyValue tests that empty values are handled as empty strings.
func TestParseEmptyValue(t *testing.T) {
	input := `name =`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	entry, ok := doc.Get("name")
	if !ok {
		t.Fatal("key 'name' not found")
	}
	if entry.Value != "" {
		t.Errorf("expected empty string, got %q", entry.Value)
	}
}

// TestParseErrors tests that invalid input produces errors.
func TestParseErrors(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"missing equals", "this has no separator"},
		{"invalid key start", "123key = value"},
		{"empty key", " = value"},
		{"key with space", "bad key = value"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(tt.input)
			if err == nil {
				t.Error("expected error but got nil")
			}
		})
	}
}

// TestParseCompleteDocument tests a realistic configuration document.
func TestParseCompleteDocument(t *testing.T) {
	input := `
# Application configuration
app.name = "My Service"
app.version = "1.2.3"
app.debug = false

# Server settings
server.host = "0.0.0.0"
server.port = 8080
server.timeout = 30

# Database
db.host = "localhost"
db.port = 5432
db.max_connections = 25
db.ssl_enabled = true
`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if doc.Len() != 10 {
		t.Errorf("expected 10 entries, got %d", doc.Len())
	}

	// Spot-check a few values
	entry, ok := doc.Get("server.port")
	if !ok {
		t.Fatal("key 'server.port' not found")
	}
	if entry.Value != "8080" || entry.Type != TypeInteger {
		t.Errorf("unexpected server.port: %+v", entry)
	}
}

// TestParseUnterminatedString tests that unterminated strings produce an error.
func TestParseUnterminatedString(t *testing.T) {
	input := `name = "missing end quote`

	_, err := Parse(input)
	if err == nil {
		t.Error("expected error for unterminated string, got nil")
	}
}

// TestDocumentOverwrite tests that duplicate keys are overwritten.
func TestDocumentOverwrite(t *testing.T) {
	input := `
name = "first"
name = "second"
`

	doc, err := Parse(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	entry, ok := doc.Get("name")
	if !ok {
		t.Fatal("key 'name' not found")
	}
	if entry.Value != "second" {
		t.Errorf("expected 'second', got %q", entry.Value)
	}
}
