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
