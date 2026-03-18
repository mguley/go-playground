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
