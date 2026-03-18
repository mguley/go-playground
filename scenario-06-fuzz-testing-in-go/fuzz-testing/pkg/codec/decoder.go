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
