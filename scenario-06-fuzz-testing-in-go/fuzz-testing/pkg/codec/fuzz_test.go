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
