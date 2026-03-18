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
