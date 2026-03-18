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
