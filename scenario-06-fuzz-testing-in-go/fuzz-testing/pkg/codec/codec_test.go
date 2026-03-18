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
