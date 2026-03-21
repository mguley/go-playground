# Fuzz testing and property-based testing in Go

## Table of Contents
- [Introduction](#introduction)
- [What is fuzz testing?](#what-is-fuzz-testing)
- [Fuzz testing vs. property-based testing](#fuzz-testing-vs-property-based-testing)
- [Prerequisites](#prerequisites)
- [Step 1: Understanding our domain - a lightweight data format](#step-1-understanding-our-domain---a-lightweight-data-format)
- [Step 2: Building the parser](#step-2-building-the-parser)
- [Step 3: Traditional tests and their blind spots](#step-3-traditional-tests-and-their-blind-spots)
- [Step 4: Your first fuzz test - crash resistance](#step-4-your-first-fuzz-test---crash-resistance)
- [Step 5: Interpreting and fixing fuzz findings](#step-5-interpreting-and-fixing-fuzz-findings)
- [Step 6: Building the binary codec](#step-6-building-the-binary-codec)
- [Step 7: Property-based roundtrip testing](#step-7-property-based-roundtrip-testing)
- [Step 8: Advanced fuzzing - testing structural invariants](#step-8-advanced-fuzzing---testing-structural-invariants)
- [Conclusion](#conclusion)

---

#### Introduction

Picture this: you have written a parser for a configuration file format. You have 95% test coverage.
You have tested quoted strings, integers, booleans, comments, edge cases with empty values, and error paths for malformed input. Every test passes. You ship it to production.

Three weeks later, a customer uploads a configuration file that was edited on Windows.
Your parser panics. The panic crashes the HTTP handler, which in turn crashes the goroutine, which manifests as a 500 error that triggers a cascade of retries from upstream services.
Your on-call engineer spends two hours at 2 AM tracing the issue to a single line in your parser where a backslash at the end of a quoted string causes an index out-of-bounds access.

The input that caused the crash was `key = "path\\"` - a perfectly reasonable configuration value representing a Windows file path.
Your unit tests never tried this input because no human thought to include it. And therein lies the fundamental limitation of example-based testing: **you can only test the inputs you think of.**

Fuzz testing flips this model on its head. Instead of you providing specific inputs, the fuzzer generates thousands or millions of inputs automatically, guided by code coverage feedback, probing paths through your code that no human would think to explore.
When it finds an input that causes a crash or violates an expected property, it minimizes it to the smallest possible reproducing case and saves it for you to examine.

Go has had first-class fuzz testing built into its standard testing framework since version 1.18.
This is unusual among mainstream languages - most require third-party tools for fuzzing.
In Go, fuzz testing is as natural as writing a unit test: you add a function starting with `Fuzz` in a `_test.go` file and run it with `go test -fuzz`.

In this deep dive, we will build a lightweight key-value data format from scratch - a text parser and a binary codec - and use fuzz testing to discover defects that our carefully crafted unit tests miss entirely.
You will experience firsthand the moment when the fuzzer finds a crash in code you were confident was correct, and you will learn the techniques that make fuzz testing effective in production Go code.

---

#### What is fuzz testing?

Fuzz testing (or "fuzzing") is an automated testing technique that feeds semi-random inputs to a program, monitoring for crashes, assertion failures, or other unexpected behavior.
The idea is simple: if your code claims to handle arbitrary input gracefully, let a machine prove it by throwing everything it can think of at your code.

Modern fuzzers like the one built into Go are **coverage-guided**. This means they don't just generate purely random bytes.
Instead, they instrument your code to track which branches and code paths are executed for each input.
When the fuzzer discovers an input that reaches a new branch, it saves that input and uses it as a seed for further mutations.
Over time, the fuzzer builds up a collection of inputs (called a **corpus**) that collectively exercise an increasing proportion of your code's logic.

This coverage guidance is what makes fuzzing dramatically more effective than random testing.
A purely random string generator would take an astronomically long time to produce a valid key-value pair like `name = "hello"`.
But a coverage-guided fuzzer, starting from seed inputs that include this pattern, can mutate the string character by character, observe which mutations reach new code paths,
and converge on interesting variations like `name = "hello\` (which reaches the escape handling code but with a trailing backslash that was never tested).

**What fuzzing finds best**

- Fuzzing excels at finding **crashes** (panics, segfaults, index-out-of-bounds), **resource exhaustion** (infinite loops, exponential blowup, out-of-memory allocations),
  **logic errors** where specific input combinations violate expected invariants, and **security vulnerabilities** in code that processes untrusted input such as parsers, decoders, deserializers, and validators.

**What fuzzing does not replace**

- Fuzz testing is not a replacement for unit tests or integration tests.
  It does not verify business logic ("when a user clicks checkout, the order should be created"), it does not test interactions between components, and it does not validate that your API returns the correct HTTP status codes.
  Fuzz testing complements traditional testing, specifically targeting defects caused by unexpected inputs.

---

#### Fuzz testing vs. property-based testing

Fuzz testing and property-based testing are closely related but emphasize different aspects of automated testing.

- **Fuzz testing** focuses on **finding inputs that crash your code**. The typical fuzz test looks like: "feed arbitrary bytes to my parser and verify it never panics."
  The property being tested is simple - crash resistance - and the emphasis is on exploring the input space as broadly as possible.

- **Property-based testing** focuses on **verifying that properties hold across all inputs**.
  A property-based test looks like: "for any valid document, encoding it to binary and decoding it back produces an equivalent document."
  The emphasis is on the property being checked, and the random input generation is the mechanism for checking it.

In practice, the boundary between these techniques is fuzzy. Go's built-in fuzzing framework supports both approaches.
- A fuzz target that simply calls a function and checks for panics is pure fuzz testing.
- A fuzz target that generates structured inputs, performs an operation, and checks that invariants hold is a property-based testing approach.
  We will use both approaches in this scenario.

The most common properties to test are:

- **Crash resistance**: any input should produce either a valid result or an error, never a panic. This is the most basic property and applies to any function that accepts external input.

- **Roundtrip consistency**: encoding then decoding (or vice versa) should produce the original value. This applies to serializers, compressors, encryptors, or any pair of inverse functions. The formal way to state this is: `decode(encode(x)) == x` for all valid `x`.

- **Idempotency**: applying an operation twice produces the same result as applying it once. This is relevant for formatters, normalizers, and operations that claim to be idempotent.

- **Invariant preservation**: certain structural properties should hold regardless of the input. For example, a sorted list should remain sorted after insertion, a balanced tree should remain balanced after rotation, or a parsed document should have the same number of entries as there are key-value lines in the input.

---

#### Prerequisites

Before we begin, ensure you have the following:

- Go 1.26 or later
- A code editor of your choice
- Basic understanding of Go testing with the `testing` package
- Familiarity with writing `Test` functions and running `go test`
- A terminal for running commands

---

#### Step 1: Understanding our domain - a lightweight data format

Before writing any code, let us understand what we are building and why it makes a good target for fuzz testing.

Our domain is a lightweight key-value data format, similar in spirit to INI files, `.env` files, or simplified TOML.
This is the kind of format that appears everywhere in software: configuration files, HTTP headers, environment variable loaders, and protocol metadata.
The format is simple enough to implement in under 200 lines of Go, but complex enough to harbor subtle defects.

**The text format**

```
# This is a comment
app.name = "My Service"
app.port = 8080
app.debug = false
app.ratio = 0.75
```

The rules are straightforward. Lines starting with `#` are comments and are ignored. Empty lines are ignored. Everything else is a key-value pair separated by `=`.
Keys must start with a letter or underscore and can contain letters, digits, underscores, and dots.
Values can be quoted strings (supporting escape sequences `\"`, `\\`, `\n`, `\t`), integers, floating-point numbers, or booleans (`true`/`false`).
Unquoted values that do not match any specific type are treated as plain strings.

**The binary codec**

In addition to the text format, we will build a binary encoder and decoder for the same data.
The binary format uses a compact wire representation with length-prefixed fields, type tags, and fixed-width numeric encodings.
This second component gives us a rich surface for roundtrip property testing: parse text into a document, encode it to binary, decode it back, and verify the result matches the original.

**Why this is a good fuzzing target**

Parsers and codecs are ideal fuzzing targets because they accept arbitrary input (often from untrusted sources), they have complex control flow with many branches,
they perform string manipulation and byte-level operations that are prone to off-by-one errors, and they are expected to handle malformed input gracefully rather than crashing.
A parser that panics on unexpected input is a security vulnerability if that input comes from a network connection or a user-uploaded file.

Let us create our project structure and initialize the module:

```bash
mkdir -p fuzz-testing/{pkg/parser,pkg/codec}
cd fuzz-testing
go mod init fuzz-testing
```

Your `go.mod` file should look like this:

```
module fuzz-testing

go 1.26.0
```

---

#### Step 2: Building the parser

Let us build the text format parser. We will start with the data types, then implement the parser itself.
The code we write here will contain a subtle defect that our unit tests in `Step 3` will not catch, but our fuzz test in `Step 4` will discover.

First, let us define the core data structures.

Create `pkg/parser/types.go`:

```go
package parser

import (
	"fmt"
	"strings"
)

// ValueType represents the type of a value in our key-value format.
// Our format supports four types: strings, integers, floating-point
// numbers, and booleans. The parser infers the type from the syntax
// of the value (quoted strings, numeric literals, true/false keywords).
type ValueType uint8

const (
	// TypeString represents a string value. Strings can be either
	// quoted ("hello world") or unquoted (hello). Quoted strings
	// support escape sequences: \", \\, \n, \t.
	TypeString ValueType = iota

	// TypeInteger represents an integer value. Integers are sequences
	// of digits optionally preceded by a minus sign: 42, -17, 0.
	TypeInteger

	// TypeFloat represents a floating-point value. Floats contain
	// a decimal point: 3.14, -0.5, 100.0.
	TypeFloat

	// TypeBoolean represents a boolean value: true or false.
	TypeBoolean
)

// String returns the human-readable name of a ValueType.
func (t ValueType) String() string {
	switch t {
	case TypeString:
		return "string"
	case TypeInteger:
		return "integer"
	case TypeFloat:
		return "float"
	case TypeBoolean:
		return "boolean"
	default:
		return fmt.Sprintf("unknown(%d)", t)
	}
}

// Entry represents a single key-value pair in a document.
// The Key is always a string, while the Value can be any of the
// supported types. The Raw field preserves the original string
// representation from the source text, which is useful for
// round-trip fidelity and debugging.
type Entry struct {
	Key   string    `json:"key"`
	Value string    `json:"value"` // The interpreted value as a string
	Type  ValueType `json:"type"`
	Raw   string    `json:"raw"` // The original text representation
}

// Document represents a parsed key-value document. It maintains
// insertion order of entries and provides key-based lookup.
type Document struct {
	Entries []Entry
	index   map[string]int // maps key to index in Entries slice
}

// NewDocument creates an empty document.
func NewDocument() *Document {
	return &Document{
		Entries: make([]Entry, 0),
		index:   make(map[string]int),
	}
}

// Add appends an entry to the document. If a key already exists,
// the previous entry is overwritten.
func (d *Document) Add(entry Entry) {
	if idx, exists := d.index[entry.Key]; exists {
		d.Entries[idx] = entry
		return
	}
	d.index[entry.Key] = len(d.Entries)
	d.Entries = append(d.Entries, entry)
}

// Get retrieves an entry by key. Returns the entry and true if found,
// or a zero Entry and false if not found.
func (d *Document) Get(key string) (Entry, bool) {
	if idx, exists := d.index[key]; exists {
		return d.Entries[idx], true
	}
	return Entry{}, false
}

// Len returns the number of entries in the document.
func (d *Document) Len() int {
	return len(d.Entries)
}

// Equal compares two documents for equality. Two documents are equal
// if they contain the same entries in the same order with the same
// keys, values, and types.
func (d *Document) Equal(other *Document) bool {
	if d.Len() != other.Len() {
		return false
	}
	for i, entry := range d.Entries {
		otherEntry := other.Entries[i]
		if entry.Key != otherEntry.Key {
			return false
		}
		if entry.Value != otherEntry.Value {
			return false
		}
		if entry.Type != otherEntry.Type {
			return false
		}
	}
	return true
}

// String returns a human-readable representation of the document.
func (d *Document) String() string {
	var sb strings.Builder
	for _, entry := range d.Entries {
		sb.WriteString(fmt.Sprintf("%s = %s (%s)\n", entry.Key, entry.Value, entry.Type))
	}
	return sb.String()
}
```

Now let us implement the parser. Read through this code carefully - there is a defect hiding in plain sight.
It passes every unit test we will write in the next step, but the fuzzer will find it.

Create `pkg/parser/parser.go`:

```go
package parser

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Parse reads a string in our key-value format and returns a Document.
//
// The format supports:
//   - Comments: lines starting with # are ignored
//   - Empty lines: ignored
//   - Key-value pairs: key = value
//   - Quoted strings: key = "hello world"
//   - Integers: key = 42
//   - Floats: key = 3.14
//   - Booleans: key = true | false
//
// Keys must start with a letter or underscore and can contain letters,
// digits, underscores, and dots (for hierarchical keys like database.host).
func Parse(input string) (*Document, error) {
	doc := NewDocument()
	lines := strings.Split(input, "\n")

	for lineNum, line := range lines {
		// Trim whitespace from the line
		line = strings.TrimSpace(line)

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Split on the first '=' to separate key and value
		eqIdx := strings.Index(line, "=")
		if eqIdx == -1 {
			return nil, fmt.Errorf("line %d: missing '=' separator in: %q", lineNum+1, line)
		}

		key := strings.TrimSpace(line[:eqIdx])
		rawValue := strings.TrimSpace(line[eqIdx+1:])

		// Validate the key
		if err := validateKey(key); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum+1, err)
		}

		// Parse the value based on its syntax
		entry, err := parseValue(key, rawValue)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum+1, err)
		}

		doc.Add(entry)
	}

	return doc, nil
}

// validateKey checks that a key follows our naming rules:
// must start with a letter or underscore, can contain letters,
// digits, underscores, and dots.
func validateKey(key string) error {
	if key == "" {
		return fmt.Errorf("empty key")
	}

	for i, ch := range key {
		if i == 0 {
			if !unicode.IsLetter(ch) && ch != '_' {
				return fmt.Errorf("key must start with a letter or underscore, got %q", ch)
			}
			continue
		}
		if !unicode.IsLetter(ch) && !unicode.IsDigit(ch) && ch != '_' && ch != '.' {
			return fmt.Errorf("invalid character %q in key %q", ch, key)
		}
	}
	return nil
}

// parseValue examines the raw value string and determines its type.
// Quoted strings start with ", booleans are true/false keywords,
// and numeric values are detected by attempting to parse them.
func parseValue(key, raw string) (Entry, error) {
	if raw == "" {
		// Empty value is treated as an empty string
		return Entry{Key: key, Value: "", Type: TypeString, Raw: raw}, nil
	}

	// Check for quoted string
	if raw[0] == '"' {
		value, err := parseQuotedString(raw)
		if err != nil {
			return Entry{}, fmt.Errorf("key %q: %w", key, err)
		}
		return Entry{Key: key, Value: value, Type: TypeString, Raw: raw}, nil
	}

	// Check for boolean
	lower := strings.ToLower(raw)
	if lower == "true" || lower == "false" {
		return Entry{Key: key, Value: lower, Type: TypeBoolean, Raw: raw}, nil
	}

	// Check for integer (must not contain a decimal point)
	if !strings.Contains(raw, ".") {
		if _, err := strconv.ParseInt(raw, 10, 64); err == nil {
			return Entry{Key: key, Value: raw, Type: TypeInteger, Raw: raw}, nil
		}
	}

	// Check for float
	if _, err := strconv.ParseFloat(raw, 64); err == nil {
		return Entry{Key: key, Value: raw, Type: TypeFloat, Raw: raw}, nil
	}

	// Default: treat as unquoted string
	return Entry{Key: key, Value: raw, Type: TypeString, Raw: raw}, nil
}

// parseQuotedString parses a quoted string value, handling escape sequences.
// The input must start with a double quote. Supported escapes: \", \\, \n, \t.
//
// This function contains a subtle defect that is not caught by typical unit tests
// but is discovered by fuzz testing. Can you spot it before reading on?
func parseQuotedString(s string) (string, error) {
	if len(s) < 2 || s[0] != '"' {
		return "", fmt.Errorf("expected opening quote")
	}

	var buf strings.Builder
	i := 1 // start after the opening quote

	for i < len(s) {
		ch := s[i]

		if ch == '\\' {
			// Handle escape sequence: advance past the backslash
			// and interpret the next character.
			i++
			switch s[i] {
			case '"':
				buf.WriteByte('"')
			case '\\':
				buf.WriteByte('\\')
			case 'n':
				buf.WriteByte('\n')
			case 't':
				buf.WriteByte('\t')
			default:
				// Unknown escape: preserve both characters
				buf.WriteByte('\\')
				buf.WriteByte(s[i])
			}
			i++
			continue
		}

		if ch == '"' {
			// Found the closing quote
			return buf.String(), nil
		}

		buf.WriteByte(ch)
		i++
	}

	return "", fmt.Errorf("unterminated string: missing closing quote")
}
```

Take a moment to examine the `parseQuotedString` function. The code looks clean and handles several cases: normal characters are appended to the buffer,
escape sequences are decoded, the closing quote terminates the string, and reaching the end without a closing quote returns an error.
Every meaningful code path appears to be covered.

But there is a defect. We will let the fuzzer find it.

---

#### Step 3: Traditional tests and their blind spots

Now let us write comprehensive unit tests for our parser.
These tests cover a wide range of scenarios: basic key-value pairs, all four value types, escape sequences, comments, dotted keys, empty values, and various error conditions.
They represent the kind of thorough test suite that a diligent engineer would write.

Create `pkg/parser/parser_test.go`:

```go
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
```

Run the tests:

```bash
go test -v ./pkg/parser/
```

Every test passes. We have tested strings, integers, floats, booleans, comments, dotted keys, empty values, escapes, unterminated strings, error conditions, and a realistic multi-entry document.
This is a solid test suite by conventional standards.

And yet, our parser has a defect.

The fundamental problem with example-based testing is that it can only verify the inputs you think of. Our escape sequence tests cover `\"`, `\\`, `\n`, and `\t`.
But we never tested a backslash as the **last character** of an unterminated string. We never tested a backslash followed by an end-of-string.
We never tested many thousands of other combinations that could exercise the escape handling code in unexpected ways.

This is exactly the gap that fuzz testing fills.

---

#### Step 4: Your first fuzz test - crash resistance

Go's fuzz testing is integrated directly into the `testing` package. A fuzz test looks similar to a regular test but uses `*testing.F` instead of `*testing.T`, starts with `Fuzz` instead of `Test`, and has two phases: seeding and fuzzing.

- In the **seed phase**, you provide initial inputs using `f.Add(...)`. These seeds serve two purposes: they define the types of the fuzz function's parameters (Go infers the types from the seed values), and they give the fuzzer a starting point that it can mutate to explore the input space.
  Good seeds cover different code paths and include both valid and invalid inputs.

- In the **fuzz phase**, the fuzzer takes your seeds, mutates them using coverage-guided algorithms, and calls your fuzz function with the mutated inputs.
  The fuzz function should check properties that must hold for all inputs. If the function panics or calls `t.Fatal`/`t.Error`, the fuzzer records the failing input and stops.

Let us write our first fuzz test.

Create `pkg/parser/fuzz_test.go`:

```go
package parser

import (
	"testing"
	"unicode/utf8"
)

// FuzzParse is our first fuzz test. It feeds random strings to the parser
// and verifies a fundamental property: the parser must never panic,
// regardless of what input it receives. A well-written parser should
// either return a valid Document or return an error, but it should
// never crash with an unrecovered panic.
//
// This is the "crash resistance" property, and it is the most basic
// property any parser should satisfy.
//
// Run with:
//
//	go test -fuzz=FuzzParse -fuzztime=30s ./pkg/parser/
func FuzzParse(f *testing.F) {
	// Seed the corpus with examples that cover different code paths.
	// These seeds guide the fuzzer toward interesting input regions.
	// Without seeds, the fuzzer starts from truly random bytes and takes
	// longer to discover structured patterns like key=value pairs.
	f.Add(`name = "hello"`)
	f.Add(`count = 42`)
	f.Add(`ratio = 3.14`)
	f.Add(`enabled = true`)
	f.Add(`# comment`)
	f.Add(``)
	f.Add(`key = "escaped \"quote\""`)
	f.Add(`key = "backslash \\"`)
	f.Add(`key = "tab\there"`)
	f.Add("key = \"newline\\nhere\"")
	f.Add("multiline\nkey1 = value1\nkey2 = value2")
	f.Add(`no_separator_here`)
	f.Add(` = value_without_key`)
	f.Add(`key =`)
	f.Add(`key = "unterminated`)

	f.Fuzz(func(t *testing.T, input string) {
		// The only property we check: Parse must not panic.
		// It may return an error (that is fine and expected for
		// malformed input), or it may return a valid document.
		doc, err := Parse(input)

		if err != nil {
			// Parser correctly rejected the input. Nothing more to check.
			return
		}

		// If parsing succeeded, verify some basic structural properties
		// of the result. These are invariants that must hold for any
		// successfully parsed document.

		if doc == nil {
			t.Fatal("Parse returned nil document without an error")
		}

		for i, entry := range doc.Entries {
			// Every entry must have a non-empty key
			if entry.Key == "" {
				t.Errorf("entry %d has empty key", i)
			}

			// The key must be valid UTF-8
			if !utf8.ValidString(entry.Key) {
				t.Errorf("entry %d key is not valid UTF-8: %q", i, entry.Key)
			}

			// The type must be one of our known types
			if entry.Type > TypeBoolean {
				t.Errorf("entry %d has unknown type: %d", i, entry.Type)
			}
		}
	})
}
```

Now let us run the fuzzer:

```bash
go test -fuzz=FuzzParse -fuzztime=10s ./pkg/parser/
```

Within seconds, you should see output similar to this:

```
fuzz: elapsed: 0s, gathering baseline coverage: 0/21 completed
fuzz: elapsed: 0s, gathering baseline coverage: 21/21 completed, now fuzzing with 24 workers
fuzz: elapsed: 0s, execs: 1429 (8591/sec), new interesting: 7 (total: 28)
--- FAIL: FuzzParse (0.17s)
    --- FAIL: FuzzParse (0.00s)
        testing.go:1927: panic: runtime error: index out of range [2] with length 2
            goroutine 204 [running]:
            ........
```

The fuzzer found a crash. In under a second. Our 15 carefully crafted unit tests all passed, but the fuzzer found the defect almost immediately.

---

#### Step 5: Interpreting and fixing fuzz findings

When the fuzzer finds a crashing input, it saves the input to a file in the `testdata/fuzz/<FuzzTestName>` directory.
This file serves two purposes: it lets you examine what caused the crash, and it becomes a permanent regression test that runs on every future go test invocation.

Let us examine what the fuzzer found. List the corpus directory:

```bash
ls pkg/parser/testdata/fuzz/FuzzParse/
```

You will see a file with a hash-based name (example):

```bash
bf621760a0561c2c
```

View its contents:

```bash
cat pkg/parser/testdata/fuzz/FuzzParse/*
```

The file contains something like:

```
go test fuzz v1
string("A=\"\\")
```

**Decoding the corpus file format**

The corpus file uses Go string literal escaping, which can be confusing at first glance.
Let us decode `string("A=\"\\")` step by step. Inside the double quotes, every backslash-prefixed pair is a single character:
- `A` -> literal `A`
- `=` -> literal `=`
- `\"` -> literal `"` (escaped double quote)
- `\\` -> literal `\` (escaped backslash)

So the actual string the fuzzer discovered is four characters: `A="\`.

This is a key-value pair where the key is `A` and the value begins with a double quote followed by a backslash, but the string never closes.
Let's be honest that we do not always think to test this exact input (or similar cases), but the fuzzer found it almost immediately by mutating the seeds we provided.

**Tracing the crash**

Let us walk through exactly what the parser does with the input `A="\`.

First, `Parse` processes the single line `A="\`. It finds the `=` at index 1, splits to get key `A` and raw value `"\` (two characters: a double quote and a backslash).
Key validation passes (single letter), and then `parseValue` is called. Since the raw value starts with `"`, it calls `parseQuotedString("\")`.

Now we are inside `parseQuotedString` with `s = "\`, which has length 2:

1. The guard `len(s) < 2` evaluates to false (2 is not less than 2), and `s[0]` is `"`, so we pass the opening check.
2. `i` starts at 1 (the position after the opening quote). We enter the loop because `i < len(s)` (1 < 2).
3. `s[1]` is `\`, so `ch` equals `'\\'`. We enter the escape-handling branch.
4. The code executes `i++`, advancing `i` to 2.
5. The very next line executes `switch s[i]` - but `i` is 2 and the string length is 2. Index 2 is out of bounds. `Panic: index out of range [2] with length 2`.

This matches the crash output from the terminal precisely:

```bash
panic: runtime error: index out of range [2] with length 2
```

The root cause is that after incrementing `i` past the backslash, the code assumes there is always a next character to read. When the backslash is the last character in the string, that assumption is violated.
This is a classic off-by-one / bounds-check error, and it is the single most common class of defect that fuzzers find in parsers.

**Why our unit tests missed this**

Our escape sequence tests all used well-formed inputs where every backslash was followed by a recognized escape character (`\"`, `\\`, `\n`, `\t`), and the string was always properly terminated with a closing quote.
We also tested an unterminated string (`"missing end quote`), but that string contained no backslash, so it hit the "unterminated string: missing closing quote" error path cleanly without touching the escape handler.

The specific combination - a backslash as the very last character with no character following it - falls into a gap between these two test cases.
It enters the escape handler (because there is a backslash) but runs out of characters before the handler can read what comes after the backslash.

**Applying the fix**

The fix is simple: after incrementing `i` past the backslash, check that `i` is still within bounds before accessing `s[i]`.

```go
if ch == '\\' {
    i++
    if i >= len(s) {
        return "", fmt.Errorf("unterminated escape sequence at end of string")
    }
    switch s[i] {
    // ... cases unchanged
```

Update the `parseQuotedString` function in `pkg/parser/parser.go`:

```go
func parseQuotedString(s string) (string, error) {
	if len(s) < 2 || s[0] != '"' {
		return "", fmt.Errorf("expected opening quote")
	}

	var buf strings.Builder
	i := 1 // start after the opening quote

	for i < len(s) {
		ch := s[i]

		if ch == '\\' {
			i++
			// FIX: Check bounds after advancing past the backslash.
			// Without this check, a backslash as the last character
			// causes an index-out-of-range panic. This defect was found
			// by fuzz testing in under one second.
			if i >= len(s) {
				return "", fmt.Errorf("unterminated escape sequence")
			}
			switch s[i] {
			case '"':
				buf.WriteByte('"')
			case '\\':
				buf.WriteByte('\\')
			case 'n':
				buf.WriteByte('\n')
			case 't':
				buf.WriteByte('\t')
			default:
				buf.WriteByte('\\')
				buf.WriteByte(s[i])
			}
			i++
			continue
		}

		if ch == '"' {
			return buf.String(), nil
		}

		buf.WriteByte(ch)
		i++
	}

	return "", fmt.Errorf("unterminated string: missing closing quote")
}
```

The added lines are the `if i >= len(s)` check immediately after `i++`.
If the backslash was the last character, `i` now equals `len(s)`, the check triggers, and we return a descriptive error instead of panicking.

**Verifying the fix**

First, let us confirm that the saved corpus entry - which previously caused the crash - now produces a clean error:

```bash
go test -run=FuzzParse ./pkg/parser/
```

This runs the `FuzzParse` function against all entries in its seed corpus (both the `f.Add()` seeds and the saved crash file `bf621760a0561c2c`) without generating new inputs.
If the fix is correct, all entries pass: the crash input `A="\` now triggers the `"unterminated escape sequence"` error path instead of panicking, and the fuzz function treats returned errors as acceptable outcomes.

Now run the fuzzer again to search for additional crashes:

```bash
go test -fuzz=FuzzParse -fuzztime=30s ./pkg/parser/
```

This time, the fuzzer should run for the full 30 seconds without finding any new crashes:

```bash
fuzz: elapsed: 0s, gathering baseline coverage: 0/34 completed
fuzz: elapsed: 0s, gathering baseline coverage: 34/34 completed, now fuzzing with 24 workers
fuzz: elapsed: 3s, execs: 331843 (110586/sec), new interesting: 145 (total: 179)
fuzz: elapsed: 6s, execs: 650354 (106192/sec), new interesting: 171 (total: 205)
fuzz: elapsed: 9s, execs: 1052661 (134015/sec), new interesting: 198 (total: 232)
fuzz: elapsed: 12s, execs: 1559903 (169152/sec), new interesting: 225 (total: 259)
fuzz: elapsed: 15s, execs: 2114348 (184815/sec), new interesting: 237 (total: 271)
fuzz: elapsed: 18s, execs: 2590697 (158759/sec), new interesting: 246 (total: 280)
fuzz: elapsed: 21s, execs: 2984071 (131166/sec), new interesting: 259 (total: 293)
fuzz: elapsed: 24s, execs: 3374960 (130247/sec), new interesting: 270 (total: 304)
fuzz: elapsed: 27s, execs: 3803032 (142564/sec), new interesting: 284 (total: 318)
fuzz: elapsed: 30s, execs: 4195468 (130990/sec), new interesting: 290 (total: 324)
fuzz: elapsed: 31s, execs: 4195468 (0/sec), new interesting: 290 (total: 324)
PASS
ok  	fuzz-testing/pkg/parser	31.038s
```

Four million inputs tested, zero crashes. The fix works.

**The crashing input is now a permanent regression test**.

The file `bf621760a0561c2c` saved in `testdata/fuzz/FuzzParse/` will be included in every future `go test` run as a regular test case, ensuring this specific defect never reappears.
This is one of the most valuable aspects of fuzz testing: every defect it finds becomes an automatic regression test.
You do not need to manually write a test case for the discovered input - it is already there in the corpus, and the Go test runner knows to replay it.

---

#### Step 6: Building the binary codec

Now let us build the binary codec - an encoder that serializes `Documents` to bytes and a decoder that reads them back.
This gives us a second fuzzing target and, more importantly, enables the powerful roundtrip property test.

**The wire format**

Our binary format uses a simple structure:

```
Header:
  [magic: 4 bytes "KVDC"] [version: 1 byte] [entry_count: 2 bytes uint16]

Each entry:
  [key_length: 2 bytes uint16] [key: N bytes] 
  [type_tag: 1 byte] 
  [value_length: 2 bytes uint16] [value: M bytes]
```

Type tags are:
- `0x00` for string
- `0x01` for integer (encoded as 8-byte big-endian int64)
- `0x02` for float (8-byte big-endian float64)
- `0x03` for boolean (1 byte, 0 or 1)

Create `pkg/codec/encoder.go`:

```go
package codec

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"fuzz-testing/pkg/parser"
)

var magic = [4]byte{'K', 'V', 'D', 'C'} // "KVDC" = Key-Value Document Codec

const (
	version       = 1
	headerSize    = 7 // 4 (magic) + 1 (version) + 2 (count)
	typeString    = 0x00
	typeInteger   = 0x01
	typeFloat     = 0x02
	typeBoolean   = 0x03
	maxKeyLength  = 65535
	maxValueBytes = 65535
)

// Encode serializes a Document into its binary wire format.
// Returns the encoded bytes or an error if the document contains
// values that cannot be represented by this format.
func Encode(doc *parser.Document) ([]byte, error) {
	if doc == nil {
		return nil, fmt.Errorf("cannot encode nil document")
	}

	if len(doc.Entries) > math.MaxUint16 {
		return nil, fmt.Errorf("too many entries: %d exceeds maximum of %d",
			len(doc.Entries), math.MaxUint16)
	}

	buf := make([]byte, 0, headerSize+len(doc.Entries)*32)

	// Write header
	buf = append(buf, magic[:]...)
	buf = append(buf, version)
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(doc.Entries)))

	// Write each entry
	for _, entry := range doc.Entries {
		entryBytes, err := encodeEntry(entry)
		if err != nil {
			return nil, fmt.Errorf("encoding key %q: %w", entry.Key, err)
		}
		buf = append(buf, entryBytes...)
	}

	return buf, nil
}

func encodeEntry(entry parser.Entry) ([]byte, error) {
	keyBytes := []byte(entry.Key)
	if len(keyBytes) > maxKeyLength {
		return nil, fmt.Errorf("key too long: %d bytes", len(keyBytes))
	}

	valueBytes, typeTag, err := encodeValue(entry)
	if err != nil {
		return nil, err
	}
	if len(valueBytes) > maxValueBytes {
		return nil, fmt.Errorf("value too long: %d bytes", len(valueBytes))
	}

	var buf []byte
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(keyBytes)))
	buf = append(buf, keyBytes...)
	buf = append(buf, typeTag)
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(valueBytes)))
	buf = append(buf, valueBytes...)

	return buf, nil
}

func encodeValue(entry parser.Entry) ([]byte, byte, error) {
	switch entry.Type {
	case parser.TypeString:
		if len(entry.Value) > maxValueBytes {
			return nil, 0, fmt.Errorf("string value too long: %d bytes", len(entry.Value))
		}
		return []byte(entry.Value), typeString, nil

	case parser.TypeInteger:
		val, err := strconv.ParseInt(entry.Value, 10, 64)
		if err != nil {
			return nil, 0, fmt.Errorf("invalid integer %q: %w", entry.Value, err)
		}
		buf := make([]byte, 8)
		binary.BigEndian.PutUint64(buf, uint64(val))
		return buf, typeInteger, nil

	case parser.TypeFloat:
		val, err := strconv.ParseFloat(entry.Value, 64)
		if err != nil {
			return nil, 0, fmt.Errorf("invalid float %q: %w", entry.Value, err)
		}
		buf := make([]byte, 8)
		binary.BigEndian.PutUint64(buf, math.Float64bits(val))
		return buf, typeFloat, nil

	case parser.TypeBoolean:
		switch entry.Value {
		case "true":
			return []byte{1}, typeBoolean, nil
		case "false":
			return []byte{0}, typeBoolean, nil
		default:
			return nil, 0, fmt.Errorf("invalid boolean %q", entry.Value)
		}

	default:
		return nil, 0, fmt.Errorf("unknown type: %d", entry.Type)
	}
}
```

Now the decoder. Like the parser, this code contains a subtle defect - a missing bounds check that will cause a panic on crafted input.

Create `pkg/codec/decoder.go`:

```go
package codec

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"fuzz-testing/pkg/parser"
)

// Decode deserializes binary data into a Document.
// Returns an error if the data is malformed, too short, or uses
// an unsupported version.
//
// This function contains a subtle defect that is not caught by typical
// unit tests (which use well-formed encoded data) but is discovered
// by fuzz testing with arbitrary byte sequences.
func Decode(data []byte) (*parser.Document, error) {
	if len(data) < headerSize {
		return nil, fmt.Errorf("data too short for header: need %d bytes, got %d",
			headerSize, len(data))
	}

	if data[0] != magic[0] || data[1] != magic[1] ||
		data[2] != magic[2] || data[3] != magic[3] {
		return nil, fmt.Errorf("invalid magic bytes: expected %q, got %q",
			magic[:], data[:4])
	}

	if data[4] != version {
		return nil, fmt.Errorf("unsupported version: %d", data[4])
	}

	entryCount := int(binary.BigEndian.Uint16(data[5:7]))
	offset := headerSize

	doc := parser.NewDocument()

	for i := 0; i < entryCount; i++ {
		entry, newOffset, err := decodeEntry(data, offset)
		if err != nil {
			return nil, fmt.Errorf("entry %d: %w", i, err)
		}
		doc.Add(entry)
		offset = newOffset
	}

	if offset != len(data) {
		return nil, fmt.Errorf("trailing data: %d extra bytes", len(data)-offset)
	}

	return doc, nil
}

func decodeEntry(data []byte, offset int) (parser.Entry, int, error) {
	// Read key length
	if offset+2 > len(data) {
		return parser.Entry{}, 0, fmt.Errorf("unexpected end of data reading key length")
	}
	keyLen := int(binary.BigEndian.Uint16(data[offset : offset+2]))
	offset += 2

	// Read key bytes
	// Defect: No check that offset+keyLen <= len(data).
	// If keyLen claims a length larger than the remaining data,
	// the slice operation on the next line will panic.
	key := string(data[offset : offset+keyLen])
	offset += keyLen

	// Read type tag
	if offset >= len(data) {
		return parser.Entry{}, 0, fmt.Errorf("unexpected end of data reading type tag")
	}
	typeTag := data[offset]
	offset++

	// Read value length
	if offset+2 > len(data) {
		return parser.Entry{}, 0, fmt.Errorf("unexpected end of data reading value length")
	}
	valueLen := int(binary.BigEndian.Uint16(data[offset : offset+2]))
	offset += 2

	// Read value bytes
	// Defect: Same missing check - if valueLen exceeds remaining data,
	// this slice operation will panic.
	valueBytes := data[offset : offset+valueLen]
	offset += valueLen

	entry, err := decodeValue(key, typeTag, valueBytes)
	if err != nil {
		return parser.Entry{}, 0, err
	}

	return entry, offset, nil
}

func decodeValue(key string, typeTag byte, valueBytes []byte) (parser.Entry, error) {
	switch typeTag {
	case typeString:
		return parser.Entry{
			Key:   key,
			Value: string(valueBytes),
			Type:  parser.TypeString,
			Raw:   fmt.Sprintf("%q", string(valueBytes)),
		}, nil

	case typeInteger:
		if len(valueBytes) != 8 {
			return parser.Entry{}, fmt.Errorf("integer value must be 8 bytes, got %d", len(valueBytes))
		}
		val := int64(binary.BigEndian.Uint64(valueBytes))
		strVal := strconv.FormatInt(val, 10)
		return parser.Entry{
			Key: key, Value: strVal, Type: parser.TypeInteger, Raw: strVal,
		}, nil

	case typeFloat:
		if len(valueBytes) != 8 {
			return parser.Entry{}, fmt.Errorf("float value must be 8 bytes, got %d", len(valueBytes))
		}
		bits := binary.BigEndian.Uint64(valueBytes)
		val := math.Float64frombits(bits)
		strVal := strconv.FormatFloat(val, 'g', -1, 64)
		return parser.Entry{
			Key: key, Value: strVal, Type: parser.TypeFloat, Raw: strVal,
		}, nil

	case typeBoolean:
		if len(valueBytes) != 1 {
			return parser.Entry{}, fmt.Errorf("boolean value must be 1 byte, got %d", len(valueBytes))
		}
		val := "false"
		if valueBytes[0] != 0 {
			val = "true"
		}
		return parser.Entry{
			Key: key, Value: val, Type: parser.TypeBoolean, Raw: val,
		}, nil

	default:
		return parser.Entry{}, fmt.Errorf("unknown type tag: 0x%02x", typeTag)
	}
}
```

The defect follows the same pattern as the parser defect: after reading a length prefix, the code immediately uses that length to slice the data without checking whether enough bytes remain.

Unit tests usually miss this because they use well-formed encoded data where the lengths are correct.
But a fuzzer feeding arbitrary bytes will quickly craft a sequence where the magic and version match but the length fields claim more data than exists.

Now write the traditional tests (which all pass) and the fuzz tests that expose the defect.
The traditional codec tests follow the same pattern as the parser tests - thoroughly testing normal cases with valid input.

Create `pkg/codec/codec_test.go`:

```go
package codec

import (
	"strings"
	"testing"

	"fuzz-testing/pkg/parser"
)

func TestEncodeDecodeString(t *testing.T) {
	doc := parser.NewDocument()
	doc.Add(parser.Entry{Key: "name", Value: "Alice", Type: parser.TypeString, Raw: `"Alice"`})
	doc.Add(parser.Entry{Key: "greeting", Value: "hello world", Type: parser.TypeString, Raw: `"hello world"`})

	encoded, err := Encode(doc)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if decoded.Len() != doc.Len() {
		t.Fatalf("expected %d entries, got %d", doc.Len(), decoded.Len())
	}

	for i, original := range doc.Entries {
		result := decoded.Entries[i]
		if result.Key != original.Key {
			t.Errorf("entry %d: key mismatch: %q vs %q", i, original.Key, result.Key)
		}
		if result.Value != original.Value {
			t.Errorf("entry %d: value mismatch: %q vs %q", i, original.Value, result.Value)
		}
	}
}

func TestEncodeDecodeInteger(t *testing.T) {
	doc := parser.NewDocument()
	doc.Add(parser.Entry{Key: "count", Value: "42", Type: parser.TypeInteger, Raw: "42"})
	doc.Add(parser.Entry{Key: "negative", Value: "-17", Type: parser.TypeInteger, Raw: "-17"})
	doc.Add(parser.Entry{Key: "zero", Value: "0", Type: parser.TypeInteger, Raw: "0"})

	encoded, err := Encode(doc)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	for i, original := range doc.Entries {
		if decoded.Entries[i].Value != original.Value {
			t.Errorf("entry %d: value mismatch: %q vs %q",
				i, original.Value, decoded.Entries[i].Value)
		}
	}
}

func TestEncodeDecodeBoolean(t *testing.T) {
	doc := parser.NewDocument()
	doc.Add(parser.Entry{Key: "enabled", Value: "true", Type: parser.TypeBoolean, Raw: "true"})
	doc.Add(parser.Entry{Key: "debug", Value: "false", Type: parser.TypeBoolean, Raw: "false"})

	encoded, err := Encode(doc)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	for i, original := range doc.Entries {
		if decoded.Entries[i].Value != original.Value {
			t.Errorf("entry %d: value mismatch", i)
		}
	}
}

func TestEncodeDecodeMixedTypes(t *testing.T) {
	doc := parser.NewDocument()
	doc.Add(parser.Entry{Key: "name", Value: "Alice", Type: parser.TypeString, Raw: `"Alice"`})
	doc.Add(parser.Entry{Key: "age", Value: "30", Type: parser.TypeInteger, Raw: "30"})
	doc.Add(parser.Entry{Key: "score", Value: "9.5", Type: parser.TypeFloat, Raw: "9.5"})
	doc.Add(parser.Entry{Key: "active", Value: "true", Type: parser.TypeBoolean, Raw: "true"})

	encoded, err := Encode(doc)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if !doc.Equal(decoded) {
		t.Errorf("documents not equal:\n  original: %s\n  decoded: %s", doc, decoded)
	}
}

func TestDecodeInvalidMagic(t *testing.T) {
	data := []byte{0x00, 0x00, 0x00, 0x00, version, 0x00, 0x00}
	_, err := Decode(data)
	if err == nil {
		t.Error("expected error for invalid magic bytes")
	}
}

func TestDecodeTooShort(t *testing.T) {
	data := []byte{'K', 'V'}
	_, err := Decode(data)
	if err == nil {
		t.Error("expected error for truncated data")
	}
}

func TestEncodeNilDocument(t *testing.T) {
	_, err := Encode(nil)
	if err == nil {
		t.Error("expected error for nil document")
	}
}

func TestEncodeEmptyDocument(t *testing.T) {
	doc := parser.NewDocument()
	encoded, err := Encode(doc)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := Decode(encoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if decoded.Len() != 0 {
		t.Errorf("expected 0 entries, got %d", decoded.Len())
	}
}

func TestEncodeRejectsInvalidBoolean(t *testing.T) {
	doc := parser.NewDocument()
	doc.Add(parser.Entry{Key: "flag", Value: "yes", Type: parser.TypeBoolean, Raw: "yes"})

	_, err := Encode(doc)
	if err == nil {
		t.Fatal("expected error for invalid boolean")
	}
}

func TestEncodeRejectsOversizedString(t *testing.T) {
	doc := parser.NewDocument()
	doc.Add(parser.Entry{
		Key:   "blob",
		Value: strings.Repeat("x", maxValueBytes+1),
		Type:  parser.TypeString,
		Raw:   `"oversized"`,
	})

	_, err := Encode(doc)
	if err == nil {
		t.Fatal("expected error for oversized string value")
	}
}
```

Run the tests - they all pass. That gives us confidence that the codec handles normal, well-formed input.
But just like with the parser, ordinary tests mostly exercise happy-path data produced by our own code.

They do not answer the harder question: `What happens if the decoder receives arbitrary bytes?`

```bash
go test -v ./pkg/codec/
```

Now create the fuzz tests.

Create `pkg/codec/fuzz_test.go`:

```go
package codec

import (
	"math"
	"strconv"
	"testing"

	"fuzz-testing/pkg/parser"
)

// FuzzDecode feeds arbitrary byte sequences to the Decode function.
// The property: Decode must never panic on any input.
//
// Run with:
//
//	go test -fuzz=FuzzDecode -fuzztime=30s ./pkg/codec/
func FuzzDecode(f *testing.F) {
	// Seed with a valid encoded document
	doc := parser.NewDocument()
	doc.Add(parser.Entry{Key: "name", Value: "test", Type: parser.TypeString, Raw: `"test"`})
	doc.Add(parser.Entry{Key: "count", Value: "42", Type: parser.TypeInteger, Raw: "42"})

	validEncoded, err := Encode(doc)
	if err != nil {
		f.Fatalf("failed to encode seed: %v", err)
	}
	f.Add(validEncoded)

	// Seed with edge cases
	emptyDoc := parser.NewDocument()
	emptyEncoded, _ := Encode(emptyDoc)
	f.Add(emptyEncoded)
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{'K', 'V', 'D', 'C', 1, 0, 1})
	f.Add([]byte{'K', 'V', 'D', 'C', 1, 0, 1, 0, 3, 'k', 'e', 'y'})

	f.Fuzz(func(t *testing.T, data []byte) {
		// The decoder must not panic on any input.
		decoded, err := Decode(data)

		if err != nil {
			return
		}

		if decoded == nil {
			t.Fatal("Decode returned nil document without error")
		}
	})
}

// FuzzCodecRoundtrip tests the roundtrip property: for any valid
// document, encode(doc) -> decode -> doc2 should give doc == doc2.
//
// Run with:
//
//	go test -fuzz=FuzzCodecRoundtrip -fuzztime=30s ./pkg/codec/
func FuzzCodecRoundtrip(f *testing.F) {
	f.Add("name", "hello world", uint8(0))
	f.Add("count", "42", uint8(1))
	f.Add("ratio", "3.14", uint8(2))
	f.Add("enabled", "true", uint8(3))
	f.Add("negative", "-100", uint8(1))
	f.Add("empty_str", "", uint8(0))
	f.Add("big_num", "9999999999", uint8(1))

	f.Fuzz(func(t *testing.T, key string, value string, typeHint uint8) {
		valueType := parser.ValueType(typeHint % 4)

		if key == "" || len(key) > 1_000 {
			return
		}

		// Validate value compatibility with type
		switch valueType {
		case parser.TypeInteger:
			if _, err := strconv.ParseInt(value, 10, 64); err != nil {
				return
			}
		case parser.TypeFloat:
			if _, err := strconv.ParseFloat(value, 64); err != nil {
				return
			}
		case parser.TypeBoolean:
			if value != "true" && value != "false" {
				return
			}
		default:
		}

		doc := parser.NewDocument()
		doc.Add(parser.Entry{Key: key, Value: value, Type: valueType, Raw: value})

		encoded, err := Encode(doc)
		if err != nil {
			return
		}

		decoded, err := Decode(encoded)
		if err != nil {
			t.Fatalf("roundtrip decode failed: %v\nkey=%q value=%q type=%s",
				err, key, value, valueType)
		}

		if decoded.Len() != 1 {
			t.Fatalf("roundtrip produced %d entries, expected 1", decoded.Len())
		}

		result := decoded.Entries[0]
		if result.Key != key {
			t.Errorf("key mismatch: %q → %q", key, result.Key)
		}
		if result.Type != valueType {
			t.Errorf("type mismatch: %s → %s", valueType, result.Type)
		}

		// For floats, compare parsed values (formatting may differ)
		if valueType == parser.TypeFloat {
			originalFloat, _ := strconv.ParseFloat(value, 64)
			resultFloat, _ := strconv.ParseFloat(result.Value, 64)

			if math.IsNaN(originalFloat) && math.IsNaN(resultFloat) {
				return
			}
			if originalFloat != resultFloat {
				t.Errorf("float value mismatch: %v → %v", originalFloat, resultFloat)
			}
		} else {
			if result.Value != value {
				t.Errorf("value mismatch: %q → %q", value, result.Value)
			}
		}
	})
}
```

Run the decoder fuzz test:

```bash
go test -fuzz=FuzzDecode -fuzztime=10s ./pkg/codec/
```

Within seconds, the fuzzer should find the missing bounds check:

```
fuzz: elapsed: 0s, gathering baseline coverage: 0/9 completed
fuzz: elapsed: 0s, gathering baseline coverage: 9/9 completed, now fuzzing with 24 workers
fuzz: elapsed: 0s, execs: 2352 (9481/sec), new interesting: 9 (total: 18)
--- FAIL: FuzzDecode (0.25s)
    --- FAIL: FuzzDecode (0.00s)
        testing.go:1927: panic: runtime error: slice bounds out of range [:12345] with capacity 48
            goroutine 277 [running]:
            .........
            
    Failing input written to testdata/fuzz/FuzzDecode/a59d3b205a075bd0
    To re-run:
    go test -run=FuzzDecode/a59d3b205a075bd0
FAIL
exit status 1
FAIL	fuzz-testing/pkg/codec	0.264s
   
```

Just like with the parser in `Step 5`, the fuzzer found a crash almost immediately. Let us examine, trace, and fix it following the same process.

**Examining the corpus file**

View the saved crash input:

```bash
cat pkg/codec/testdata/fuzz/FuzzDecode/a59d3b205a075bd0
```

Output:

```bash
go test fuzz v1
[]byte("KVDC\x010000")
```

**Decoding the corpus file format**

The corpus file uses Go byte string escaping. Let us decode `[]byte("KVDC\x010000")` byte by byte:
- `K` -> 0x4B
- `V` -> 0x56
- `D` -> 0x44
- `C` -> 0x43
- `\x01` -> 0x01 (hex escape for byte value 1)
- `0` -> 0x30 (ASCII digit zero)
- `0` -> 0x30
- `0` -> 0x30
- `0` -> 0x30

That is 9 bytes total. The fuzzer crafted a byte sequence that passes the header validation (correct magic bytes, correct version) but contains fabricated length fields that claim far more data than actually exists.

Notice how the fuzzer arrived at this input.
It started from our seed `[]byte{'K', 'V', 'D', 'C', 1, 0, 1, 0, 3, 'k', 'e', 'y'}` (a partial but structurally plausible entry) and mutated it.
By trimming bytes from the end and changing others, it converged on a minimal 9-byte input that crashes the decoder.

**Tracing the crash**

Let us walk through exactly what the decoder does with these 9 bytes.

First, `Decode` processes the header. The data has 9 bytes, which passes the `len(data) < headerSize` check (9 ≥ 7).
The first four bytes are `K`, `V`, `D`, `C` - the magic bytes match. Byte 4 is `0x01` - version matches. So far, everything looks legitimate.

Next, the decoder reads the entry count from bytes 5-6. These are `0x30` and `0x30` (ASCII `'0'` and `'0'`).
Interpreted as a big-endian `uint16`, that is `0x3030` = 12,336 in decimal.
The decoder now believes there are 12,336 entries to read and enters the loop with `offset = 7` (the byte after the header).

On the first iteration, `decodeEntry` is called. It reads the key length from bytes 7–8: again `0x30` and `0x30` = 12,336.
After reading the two length bytes, `offset` advances to 9. Now the code executes:

```go
key := string(data[offset : offset+keyLen])
```

This becomes `data[9 : 9+12336]`, which is `data[9:12345]`. But the slice only has 9 bytes. `Panic: slice bounds out of range [:12345]`.

This matches the crash output from the terminal precisely:

```bash
panic: runtime error: slice bounds out of range [:12345] with capacity 48
```

The capacity varies between runs (you might see `capacity 16` or `capacity 48` depending on how Go's internal allocator sizes the underlying array),
but the bound `12345` is always the same because it comes from the deterministic calculation `9 + 12336 = 12345`.

**Why our unit tests missed this**

All of our unit tests create documents using `Encode`, which always writes correct length prefixes that match the actual data.

The defect only manifests when the decoder receives bytes that did not originate from a well-behaved encoder - which is exactly what happens with network data, corrupted files, or malicious input.
The decoder has bounds checks for reading the length prefix itself (`if offset+2 > len(data)`) but then blindly trusts the value it read.

**Applying the fix**

The fix is to add bounds checks after reading each length field, before using it to slice the data. Update `decodeEntry` in `pkg/codec/decoder.go`:

```go
func decodeEntry(data []byte, offset int) (parser.Entry, int, error) {
	// Read key length
	if offset+2 > len(data) {
		return parser.Entry{}, 0, fmt.Errorf("unexpected end of data reading key length")
	}
    keyLen := int(binary.BigEndian.Uint16(data[offset : offset+2]))
	offset += 2

	// FIX: Validate that the claimed key length doesn't exceed the remaining data.
	// Without this check, a crafted length prefix causes an out-of-bounds slice panic.
	if offset+keyLen > len(data) {
		return parser.Entry{}, 0, fmt.Errorf("data truncated: key claims %d bytes but only %d remain",
			keyLen, len(data)-offset)
	}

	// Read key bytes
	key := string(data[offset : offset+keyLen])
	offset += keyLen

	// Read type tag
	if offset >= len(data) {
		return parser.Entry{}, 0, fmt.Errorf("unexpected end of data reading type tag")
	}
	typeTag := data[offset]
	offset++

	// Read value length
	if offset+2 > len(data) {
		return parser.Entry{}, 0, fmt.Errorf("unexpected end of data reading value length")
	}
    valueLen := int(binary.BigEndian.Uint16(data[offset : offset+2]))
	offset += 2

	// FIX: Same bounds check for the value length.
	if offset+valueLen > len(data) {
		return parser.Entry{}, 0, fmt.Errorf("data truncated: value claims %d bytes but only %d remain",
			valueLen, len(data)-offset)
	}

	// Read value bytes
	valueBytes := data[offset : offset+valueLen]
	offset += valueLen

	entry, err := decodeValue(key, typeTag, valueBytes)
	if err != nil {
		return parser.Entry{}, 0, err
	}

	return entry, offset, nil
}
```

The two added checks follow the same pattern: read the length, then verify `offset + claimedLength <= len(data)` before using that length in a slice operation.

**Verifying the fix**

First, confirm that the saved corpus entry now produces a clean error instead of a panic:

```bash
go test -run=FuzzDecode ./pkg/codec/
```

This runs `FuzzDecode` against all entries in its seed corpus (both the `f.Add()` seeds and the saved crash file `a59d3b205a075bd0`) without generating new inputs.
The crash input `KVDC\x010000` now triggers the `"data truncated: key claims 12336 bytes but only 0 remain"` error path instead of panicking.

Now run the fuzzer again to search for additional crashes:

```bash
go test -fuzz=FuzzDecode -fuzztime=30s ./pkg/codec/
```

The fuzzer should run for the full duration without finding new crashes. The fix works.

---

#### Step 7: Property-based roundtrip testing

The roundtrip fuzz test (`FuzzCodecRoundtrip`) demonstrates property-based testing.

Instead of checking for crashes, it checks a semantic property: encoding a document and then decoding it should produce an equivalent document.

This is more powerful than crash testing because it catches defects where the code does not crash but silently produces incorrect results.

Run the roundtrip test (make sure you have applied the decoder bounds-check fix from `Step 6` first):

```bash
go test -fuzz=FuzzCodecRoundtrip -fuzztime=30s ./pkg/codec/
```

At first glance, you might expect this to pass now that the decoder no longer panics. But the fuzzer finds a different kind of problem:

```
fuzz: elapsed: 0s, gathering baseline coverage: 0/15 completed
fuzz: elapsed: 0s, gathering baseline coverage: 15/15 completed, now fuzzing with 24 workers
fuzz: minimizing 63-byte failing input file
fuzz: elapsed: 0s, minimizing
--- FAIL: FuzzCodecRoundtrip (0.09s)
    --- FAIL: FuzzCodecRoundtrip (0.00s)
        fuzz_test.go:130: value mismatch: "00" → "0"
    
    Failing input written to testdata/fuzz/FuzzCodecRoundtrip/a162bb72affd9c33
    To re-run:
    go test -run=FuzzCodecRoundtrip/a162bb72affd9c33
FAIL
exit status 1
FAIL	fuzz-testing/pkg/codec	0.101s
```

This is not a panic. The test did not crash.
Instead, the fuzzer found an input where the roundtrip changed the data: the value `"00"` went in, but `"0"` came out.
The property `decode(encode(x)) == x` was violated - not through a crash, but through silent data alteration.

This is exactly the class of defect that property-based testing is designed to catch, and it is the kind of defect that crash-only fuzz testing would miss entirely.

**Examining the corpus file**

View the saved failing input:

```bash
cat pkg/codec/testdata/fuzz/FuzzCodecRoundtrip/a162bb72affd9c33
```

Output:

```
go test fuzz v1
string("0")
string("00")
byte('\u009d')
```

These three values match the fuzz target signature:

```go
func(t *testing.T, key string, value string, typeHint uint8)
```

So the failing input is:
- key = `"0"`
- value = `"00"`
- typeHint = `\u009d` (which is 157 in decimal)

Inside the fuzz function, the type is computed as:

```go
valueType := parser.ValueType(typeHint % 4)
```

Since `157 % 4 == 1`, the chosen type is `parser.TypeInteger`. So the fuzzer built a one-entry document equivalent to:

```text
key="0", value="00", type=integer
```

It then encoded this to binary, decoded it back, and found that the value changed from `"00"` to `"0"`.

**Tracing the roundtrip**

Let us follow the data through each stage to understand exactly where the information is lost.

`Encoding stage`.

The encoder receives value `"00"` with type `TypeInteger`.

```go
val, err := strconv.ParseInt(entry.Value, 10, 64)
```

This succeeds and returns numeric zero. The encoder then writes that value as an 8-byte big-endian `int64`.
At this point, the information that the original string had two characters `"00"` is already lost. The binary representation holds only the numeric value, not the string form.

`Decoding stage`.

The decoder reads the 8-byte integer value, reconstructs numeric zero, and then formats it using:

```go
strVal := strconv.FormatInt(val, 10)
```

That produces `"0"`. There is no way for the decoder to know that the original string was `"00"` because that information was discarded during encoding.

`Comparison stage`.

Our original fuzz test compared raw strings for everything except floats, so it compared:
- original: `"00"`
- decoded: `"0"`

They are not equal, so the test fails.

**Why this is not a defect in the encoder or decoder individually**

The encoder is correct. The decoder is also correct.

The real problem is that the property we wrote into the test was too strict for numeric types.
The codec is lossless at the semantic level, but not lossless at the textual formatting level for integers and floats.

This is a common pattern in serializers:
- numbers lose non-canonical formatting
- timestamps may lose original timezone spelling
- Unicode text may preserve characters but not normalization form
- structured data may preserve meaning but not whitespace or ordering

Property-based tests must check the level of equivalence the format actually guarantees.

**Why our unit tests missed this**

Our unit tests used values like `"42"`, `"-17"`, and `"0"` - all of which are already in canonical form.
No unit test included `"00"`, `"007"`, `"042"`, or any other non-canonical integer string because those representations are unusual.
A human writing tests naturally reaches for clean, canonical examples.
The fuzzer has no such bias - it generates whatever strings the coverage feedback guides it toward, and `"00"` is only one mutation away from the seed value `"0"`.

**Applying the fix**

The fix is to compare parsed integer values rather than raw strings in the roundtrip test, mirroring the approach we already use for floats.
Update the comparison in `FuzzCodecRoundtrip` in `pkg/codec/fuzz_test.go`:

```go
// FuzzCodecRoundtrip tests the roundtrip property: for any valid
// document, encode(doc) -> decode -> doc2 should give doc == doc2.
//
// Run with:
//
//	go test -fuzz=FuzzCodecRoundtrip -fuzztime=30s ./pkg/codec/
func FuzzCodecRoundtrip(f *testing.F) {
	f.Add("name", "hello world", uint8(0))
	f.Add("count", "42", uint8(1))
	f.Add("ratio", "3.14", uint8(2))
	f.Add("enabled", "true", uint8(3))
	f.Add("negative", "-100", uint8(1))
	f.Add("empty_str", "", uint8(0))
	f.Add("big_num", "9999999999", uint8(1))

	f.Fuzz(func(t *testing.T, key string, value string, typeHint uint8) {
		valueType := parser.ValueType(typeHint % 4)

		if key == "" || len(key) > 1_000 {
			return
		}

		// Only continue with values that are valid for the chosen type.
		switch valueType {
		case parser.TypeInteger:
			if _, err := strconv.ParseInt(value, 10, 64); err != nil {
				return
			}
		case parser.TypeFloat:
			if _, err := strconv.ParseFloat(value, 64); err != nil {
				return
			}
		case parser.TypeBoolean:
			if value != "true" && value != "false" {
				return
			}
		default:
		}

		doc := parser.NewDocument()
		doc.Add(parser.Entry{Key: key, Value: value, Type: valueType, Raw: value})

		encoded, err := Encode(doc)
		if err != nil {
			return
		}

		decoded, err := Decode(encoded)
		if err != nil {
			t.Fatalf("roundtrip decode failed: %v\nkey=%q value=%q type=%s",
				err, key, value, valueType)
		}

		if decoded.Len() != 1 {
			t.Fatalf("roundtrip produced %d entries, expected 1", decoded.Len())
		}

		result := decoded.Entries[0]

		if result.Key != key {
			t.Errorf("key mismatch: %q - %q", key, result.Key)
		}
		if result.Type != valueType {
			t.Errorf("type mismatch: %s - %s", valueType, result.Type)
		}

		// Compare values at the semantic level appropriate for the type.
		switch valueType {
		case parser.TypeFloat:
			originalFloat, _ := strconv.ParseFloat(value, 64)
			resultFloat, _ := strconv.ParseFloat(result.Value, 64)

			if math.IsNaN(originalFloat) && math.IsNaN(resultFloat) {
				return
			}

			if originalFloat != resultFloat {
				t.Errorf("float value mismatch: %v → %v", originalFloat, resultFloat)
			}

		case parser.TypeInteger:
			originalInt, _ := strconv.ParseInt(value, 10, 64)
			resultInt, _ := strconv.ParseInt(result.Value, 10, 64)
			if originalInt != resultInt {
				t.Errorf("integer value mismatch: %v → %v", originalInt, resultInt)
			}

		default:
			// Strings and booleans are expected to roundtrip losslessly.
			if result.Value != value {
				t.Errorf("value mismatch: %q → %q", value, result.Value)
			}
		}
	})
}
```

**Verifying the fix**

First, replay the saved corpus entry:

```bash
go test -run=FuzzCodecRoundtrip ./pkg/codec/
```

This runs the fuzz target against the seed corpus, including the failing input with value `"00"`.

The input still decodes to `"0"`, but that is now accepted because both sides parse to the same integer value.

Next, run the full fuzz test again:

```bash
go test -fuzz=FuzzCodecRoundtrip -fuzztime=30s ./pkg/codec/
```

This time the fuzzer should run for the full duration without reporting the `"00" - "0"` mismatch.
It will explore millions of combinations of keys, values, and types, verifying that each of them preserves the numeric value through the encode-decode cycle.

**The broader lesson: compare semantics, not spelling**

The `"00" → "0"` failure is a classic property-based testing lesson.

A roundtrip is only meaningful once you define what `“same”` means:

- same bytes?
- same text?
- same parsed value?
- same normalized value?
- same observable behavior?

For this codec, the correct notion of equality is semantic equality by type, not raw textual equality for all cases.

That distinction is what makes property-based testing so powerful. It forces you to state, precisely, what your software promises to preserve.

---

#### Step 8: Advanced fuzzing - testing structural invariants

Beyond crash resistance and binary roundtrip consistency, fuzz testing can verify `structural invariants` across other layers of the system.

In this step, we add a `text-level roundtrip test`:

1. parse a document from text
2. serialize it back to text
3. parse the serialized form again
4. verify that the two parsed documents are equivalent

This is a different property from `FuzzCodecRoundtrip` in Step 7. That test exercised the `binary` encoder and decoder.
This one exercises the `text` parser together with a text serializer helper.

It also exposes a very Go-specific trap: a Go string is a sequence of bytes, not necessarily valid UTF-8, but `range` over a string iterates over `Unicode code points`.
When `range` encounters invalid UTF-8, it yields `U+FFFD` and advances one byte, which can silently corrupt data if you meant to preserve raw bytes.

Add the following to the existing `pkg/codec/fuzz_test.go` file.
We keep it in the `codec` package so the test can use both the parser and the serializer helper in one place.

```go
// FuzzParseRoundtrip tests a text-level roundtrip property:
// if we parse a document, serialize it back to text, and parse
// that serialized form again, the resulting document should be
// equivalent to the original parsed document.
//
// Run with:
//
//	go test -fuzz=FuzzParseRoundtrip -fuzztime=30s ./pkg/codec/
func FuzzParseRoundtrip(f *testing.F) {
	f.Add(`name = "hello world"`)
	f.Add(`count = 42`)
	f.Add(`enabled = true`)
	f.Add(`ratio = 3.14`)
	f.Add("key = \"escaped \\\"quotes\\\"\"")

	f.Fuzz(func(t *testing.T, input string) {
		doc, err := parser.Parse(input)
		if err != nil {
			return
		}

        // Serialize back to text    
		serialized := serializeDocument(doc)

		// Parse the serialized form
		doc2, err := parser.Parse(serialized)
		if err != nil {
			t.Fatalf("roundtrip failed: serialized form is invalid:\n"+
				"input: %q\nserialized: %q\nerror: %v",
				input, serialized, err)
		}

		if !doc.Equal(doc2) {
			t.Fatalf("roundtrip mismatch:\noriginal: %s\nroundtripped: %s",
				doc, doc2)
		}
	})
}

func serializeDocument(doc *parser.Document) string {
	var sb []byte
	for _, entry := range doc.Entries {
		sb = append(sb, entry.Key...)
		sb = append(sb, " = "...)

		switch entry.Type {
		case parser.TypeString:
			sb = append(sb, '"')
			for _, ch := range entry.Value {
				switch ch {
				case '"':
					sb = append(sb, '\\', '"')
				case '\\':
					sb = append(sb, '\\', '\\')
				case '\n':
					sb = append(sb, '\\', 'n')
				case '\t':
					sb = append(sb, '\\', 't')
				default:
					sb = append(sb, string(ch)...)
				}
			}
			sb = append(sb, '"')

		default:
			sb = append(sb, entry.Value...)
		}

		sb = append(sb, '\n')
	}
	return string(sb)
}
```

Run the test:

```bash
go test -fuzz=FuzzParseRoundtrip -fuzztime=30s ./pkg/codec/
```

The fuzzer will likely find a failure quickly, but not a panic:

```
fuzz: elapsed: 0s, gathering baseline coverage: 5/5 completed, now fuzzing with 24 workers
fuzz: minimizing 54-byte failing input file
fuzz: elapsed: 0s, minimizing
--- FAIL: FuzzParseRoundtrip (0.13s)
    --- FAIL: FuzzParseRoundtrip (0.00s)
        fuzz_test.go:181: roundtrip mismatch:
            original: A = � (string)

            roundtripped: A = � (string)

    Failing input written to testdata/fuzz/FuzzParseRoundtrip/40016cd733a84512
    To re-run:
    go test -run=FuzzParseRoundtrip/40016cd733a84512
FAIL
exit status 1
FAIL	fuzz-testing/pkg/codec	0.139s
```

This is not a crash. The text roundtrip completed, but the `roundtripped parsed document` does not match the `original parsed document`.

The terminal shows `�` on both sides, which makes the output look identical at first glance. But the underlying bytes differ, and that is what the test is checking.

**Examining the corpus file**

```bash
cat pkg/codec/testdata/fuzz/FuzzParseRoundtrip/40016cd733a84512
```

Output:

```
go test fuzz v1
string("A=\xd6")
```

That input contains three bytes:
- `A` = `0x41`
- `=` = `0x3D`
- `\xd6` = `0xD6`

The parser splits on `=` and gets key `A` and raw value `\xd6`.

That final byte `0xD6` is not valid UTF-8 by itself.
The Go spec defines strings as byte sequences, and string indexing returns bytes, so storing that byte in a Go string is legal.
But ranging over the string later does UTF-8 decoding rather than raw-byte iteration. [Go spec](https://go.dev/ref/spec)

**Tracing the corruption**

The problem happens in the serializer here:

```go
for _, ch := range entry.Value {
```

That loop does `rune iteration`, not byte iteration. For strings, `range` walks Unicode code points.
On invalid UTF-8, the spec says it yields `0xFFFD`, the Unicode replacement character, and advances by one byte.

So when the serializer sees the single invalid byte `0xD6`, it does not get the raw byte back. It gets the rune `U+FFFD`.

Then this line:

```go
sb = append(sb, string(ch)...)
```

writes the UTF-8 encoding of `U+FFFD`, which is the three-byte sequence:
- `0xEF`
- `0xBF`
- `0xBD`

So the original one-byte value `"\xd6"` is silently transformed into the three-byte UTF-8 sequence for `�`.

That is the same kind of invalid UTF-8 corruption shown in the official Go fuzzing tutorial: a string can contain non-UTF-8 bytes,
but converting or iterating in rune terms can replace them with `�` and lose the original data. [Go fuzzing](https://go.dev/doc/tutorial/fuzz)

**Why the terminal output is misleading**

Both the original invalid byte and the replacement character often render as `�` in a terminal, so the human-readable output looks the same.

But the actual byte content is different:
- original value: `"\xd6"` - 1 byte
- roundtripped value: `"\xef\xbf\xbd"` - 3 bytes

That is why `doc.Equal(doc2)` fails even though the printed output looks similar.

**Why this matters beyond the tutorial**

This class of defect - silent data corruption through Go's `range` over strings - appears in production code more often than you might expect.
Any time you use `for _, ch := range str` on a string that might contain non-UTF-8 bytes (binary data, filenames from certain filesystems, protocol buffers with raw byte fields, or user input with encoding errors),
you risk silently replacing bytes with U+FFFD.

The corruption is insidious because:
- the code does not panic or return an error
- the output looks plausible (the replacement character `�`is commonly seen)
- the corruption is irreversible (you cannot reconstruct 0xD6 from U+FFFD)

**Applying the fix**

The fix is to iterate over `bytes`, not runes, when serializing string values. Replace `serializeDocument` with this version:

```go
func serializeDocument(doc *parser.Document) string {
	var sb []byte
	for _, entry := range doc.Entries {
		sb = append(sb, entry.Key...)
		sb = append(sb, " = "...)

		switch entry.Type {
		case parser.TypeString:
			sb = append(sb, '"')

			// FIX: Iterate over bytes, not runes.
			// Ranging over a string decodes UTF-8 and replaces invalid
			// byte sequences with U+FFFD, which corrupts arbitrary byte data.
			for i := 0; i < len(entry.Value); i++ {
				ch := entry.Value[i]
				switch ch {
				case '"':
					sb = append(sb, '\\', '"')
				case '\\':
					sb = append(sb, '\\', '\\')
				case '\n':
					sb = append(sb, '\\', 'n')
				case '\t':
					sb = append(sb, '\\', 't')
				default:
					sb = append(sb, ch)
				}
			}

			sb = append(sb, '"')

		default:
			sb = append(sb, entry.Value...)
		}

		sb = append(sb, '\n')
	}
	return string(sb)
}
```

This works because the Go spec defines strings as sequences of bytes, and indexing a string with `s[i]` returns the byte at that position.
That preserves raw non-UTF-8 bytes exactly.

**Verifying the fix**

Replay the saved corpus entry:

```bash
go test -run=FuzzParseRoundtrip ./pkg/codec/
```

Then fuzz again:

```bash
go test -fuzz=FuzzParseRoundtrip -fuzztime=30s ./pkg/codec/
```

With the byte-iteration fix, the serializer preserves `0xD6` exactly, so the roundtripped parsed document now matches the original parsed document.

It will generate inputs containing all manner of byte sequences - multi-byte UTF-8, overlong encodings, isolated continuation bytes, null bytes, and every other combination - and the byte-level serializer handles them all correctly.

**The Go-specific lesson:** `range` over strings decodes runes

This finding highlights a Go language subtlety that is worth remembering:

```go
s := "\xd6" // one byte: 0xD6

// Byte iteration: sees the raw byte
for i := 0; i < len(s); i++ {
    fmt.Printf("%02x ", s[i]) // prints: d6
}

// Rune iteration: decodes UTF-8, replaces invalid bytes
for _, ch := range s {
    fmt.Printf("%04x ", ch)   // prints: fffd (U+FFFD replacement character)
}
```

Both loops are valid Go, but they produce different results for the same string.
`range` over a string has behavior equivalent to repeatedly decoding successive UTF-8 code points, much like `utf8.DecodeRuneInString`, including returning `U+FFFD` and advancing one byte on invalid UTF-8.

The rule of thumb: if your data might contain non-UTF-8 bytes, use byte-indexed iteration (`for i := 0; i < len(s); i++`).
If you know your data is valid UTF-8 and you want to process characters, use `range`.

---

#### Conclusion

Throughout this deep dive, we built a lightweight key-value data format from scratch - a text parser and a binary codec - and used Go's built-in fuzz testing to discover defects that our comprehensive unit test suite missed entirely.

The key takeaways from this experience are worth revisiting:

- **Traditional testing tests what you think of. Fuzz testing tests what you don't.**
  Our unit tests covered quoted strings, escape sequences, integers, booleans, error conditions, and realistic multi-entry documents.
  We had high confidence in our parser. The fuzzer found a crashing defect in under one second by generating a single input that no human thought to try: a backslash at the very end of an unterminated quoted string.
  This is not a failure of our testing discipline; it is a fundamental limitation of example-based testing.

- **The most common defects fuzzers find are boundary errors.**
  Both defects in our implementation - the parser's missing bounds check after advancing past a backslash, and the decoder's missing bounds check after reading a length prefix - are variants of the same pattern: the code reads a value that determines how many bytes to process next,
  then processes those bytes without checking that enough bytes actually exist. This pattern accounts for the vast majority of defects found by fuzzing in real-world code.

- **Property-based testing catches a different class of defects than crash testing.**
  Our crash-resistance fuzz tests (`FuzzParse`, `FuzzDecode`) verify that the code never panics.
  Our roundtrip fuzz tests (`FuzzCodecRoundtrip`, `FuzzParseRoundtrip`) verify a stronger property: that encoding and decoding are true inverses of each other.
  These properties catch silent data corruption that would not cause a crash but would produce wrong results.
  In production, silent data corruption is often more dangerous than a crash because it can propagate through your system before anyone notices.

- **Seeds guide the fuzzer, but the fuzzer's power comes from mutation.**
  Our seed inputs gave the fuzzer a starting vocabulary of valid formats and edge cases.
  The fuzzer then mutated these seeds millions of times, guided by code coverage feedback, to explore regions of the input space that our seeds never directly covered.
  The relationship between seeds and mutations is like providing a map of a city versus hiring a scout to explore every alley and rooftop - the map tells the scout where to start, but the exploration discovers what the map never could.

- **Every fuzz finding is a permanent regression test.** When the fuzzer discovers a crashing input, it saves the minimized input to your `testdata/fuzz/` directory.
  From that point forward, every `go test` run - even without the `-fuzz` flag - replays that input against your code.
  This means fuzz testing has a ratchet effect: every defect you find and fix through fuzzing makes your test suite permanently stronger.
  Over time, the accumulation of fuzz-discovered corpus entries creates a robust defense against regression.

For production use, consider extending this foundation with continuous fuzzing via OSS-Fuzz or ClusterFuzz for open-source projects, fuzz testing as a gate in your CI pipeline where PRs that introduce new panics are blocked,
structured fuzzing for protocols where you generate random but syntactically valid messages to test deeper semantic handling,
and differential fuzzing where you compare two implementations of the same specification (for instance, your parser against a reference implementation) to find behavioral differences.

Fuzz testing is not a replacement for thinking carefully about your test cases. It is a complement that covers the vast space of inputs that no human can enumerate.
The combination of thoughtful unit tests and automated fuzz testing creates a defense in depth that is far stronger than either technique alone.

Links:
- [OSS-Fuzz official docs](https://google.github.io/oss-fuzz/)
- [ClusterFuzz official docs](https://google.github.io/clusterfuzz/)
- [ClusterFuzz official GitHub repo](https://github.com/google/clusterfuzz)
- [ClusterFuzzLite official docs](https://google.github.io/clusterfuzzlite/)
