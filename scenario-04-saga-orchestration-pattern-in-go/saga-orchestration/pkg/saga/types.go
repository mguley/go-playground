package saga

import (
	"encoding/json"
	"time"
)

// SagaID uniquely identifies a saga instance. We use a string type alias
// to provide type safety while maintaining easy serialization.
type SagaID string

// StepID uniquely identifies a step within a saga definition.
type StepID string

// SagaStatus represents the current state of a saga in its lifecycle.
// See the state diagram in the README for how these states connect.
type SagaStatus string

const (
	// StatusPending indicates the saga has been created but not started.
	StatusPending SagaStatus = "PENDING"

	// StatusRunning indicates the saga is currently executing forward steps.
	StatusRunning SagaStatus = "RUNNING"

	// StatusCompleted indicates all steps completed successfully.
	// This is a terminal state.
	StatusCompleted SagaStatus = "COMPLETED"

	// StatusCompensating indicates the saga is rolling back due to a failure.
	// Compensation runs in reverse order from the last completed step.
	StatusCompensating SagaStatus = "COMPENSATING"

	// StatusCompensated indicates compensation completed successfully.
	// All side effects from completed steps have been undone.
	StatusCompensated SagaStatus = "COMPENSATED"

	// StatusFailed indicates the saga failed and could not be fully compensated.
	// This requires manual intervention to resolve the inconsistent state.
	StatusFailed SagaStatus = "FAILED"
)

// StepStatus represents the current state of an individual step within a saga.
type StepStatus string

const (
	StepPending            StepStatus = "PENDING"
	StepRunning            StepStatus = "RUNNING"
	StepCompleted          StepStatus = "COMPLETED"
	StepFailed             StepStatus = "FAILED"
	StepCompensating       StepStatus = "COMPENSATING"
	StepCompensated        StepStatus = "COMPENSATED"
	StepCompensationFailed StepStatus = "COMPENSATION_FAILED"
	StepSkipped            StepStatus = "SKIPPED"
)

// SagaState holds the complete state of a saga instance, including all
// steps and their current statuses. This state can be persisted and
// recovered to support durable saga execution across system restarts.
type SagaState struct {
	ID          SagaID         `json:"id"`
	Name        string         `json:"name"`
	Status      SagaStatus     `json:"status"`
	CurrentStep int            `json:"current_step"`
	Steps       []StepState    `json:"steps"`
	Data        map[string]any `json:"data"`
	Error       string         `json:"error,omitempty"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
	CompletedAt *time.Time     `json:"completed_at,omitempty"`
}

// StepState holds the state of an individual step within a saga.
// Each step tracks its own execution history independently.
type StepState struct {
	ID            StepID         `json:"id"`
	Name          string         `json:"name"`
	Status        StepStatus     `json:"status"`
	Result        map[string]any `json:"result,omitempty"`
	Error         string         `json:"error,omitempty"`
	RetryCount    int            `json:"retry_count"`
	StartedAt     *time.Time     `json:"started_at,omitempty"`
	CompletedAt   *time.Time     `json:"completed_at,omitempty"`
	CompensatedAt *time.Time     `json:"compensated_at,omitempty"`
}

// NewSagaState creates a new saga state with the given ID and name.
// The saga starts in PENDING status with all steps also pending.
func NewSagaState(id SagaID, name string, steps []StepState) *SagaState {
	now := time.Now()
	return &SagaState{
		ID:          id,
		Name:        name,
		Status:      StatusPending,
		CurrentStep: 0,
		Steps:       steps,
		Data:        make(map[string]any),
		CreatedAt:   now,
		UpdatedAt:   now,
	}
}

// SetData stores a value in the saga's shared data store. This data is
// available to all steps and is persisted with the saga state. Steps
// use this to pass results to subsequent steps (e.g., a reservation ID
// created in step 1 that step 2 needs to reference).
func (s *SagaState) SetData(key string, value any) {
	s.Data[key] = value
	s.UpdatedAt = time.Now()
}

// GetData retrieves a value from the saga's shared data store.
// Returns the value and a boolean indicating whether the key exists.
func (s *SagaState) GetData(key string) (any, bool) {
	val, ok := s.Data[key]
	return val, ok
}

// GetDataAs retrieves and unmarshals data into the provided target.
// This is useful when step results need to be accessed by subsequent steps
// with proper type information. The method uses JSON as an intermediate
// format to handle type conversions safely (e.g., when data was stored
// as a map[string]any but needs to be read as a struct).
//
// Returns an error if the key exists but unmarshaling fails. If the key
// does not exist, target is left unchanged and nil is returned.
func (s *SagaState) GetDataAs(key string, target any) error {
	val, ok := s.Data[key]
	if !ok {
		return nil
	}

	bytes, err := json.Marshal(val)
	if err != nil {
		return err
	}
	return json.Unmarshal(bytes, target)
}

// IsTerminal returns true if the saga is in a terminal state
// (completed, compensated, or failed). Terminal states indicate
// that the saga will not undergo any further state changes.
func (s *SagaState) IsTerminal() bool {
	return s.Status == StatusCompleted ||
		s.Status == StatusCompensated ||
		s.Status == StatusFailed
}

// Clone creates a deep copy of the saga state. This is critical
// for thread-safety when storing state - modifications to the
// original must not affect the stored copy and vice versa.
func (s *SagaState) Clone() *SagaState {
	clone := &SagaState{
		ID:          s.ID,
		Name:        s.Name,
		Status:      s.Status,
		CurrentStep: s.CurrentStep,
		Steps:       make([]StepState, len(s.Steps)),
		Data:        make(map[string]any),
		Error:       s.Error,
		CreatedAt:   s.CreatedAt,
		UpdatedAt:   s.UpdatedAt,
		CompletedAt: s.CompletedAt,
	}

	// Deep copy each step, including its Result map
	for i, step := range s.Steps {
		clone.Steps[i] = StepState{
			ID:            step.ID,
			Name:          step.Name,
			Status:        step.Status,
			Error:         step.Error,
			RetryCount:    step.RetryCount,
			StartedAt:     step.StartedAt,
			CompletedAt:   step.CompletedAt,
			CompensatedAt: step.CompensatedAt,
		}
		if step.Result != nil {
			clone.Steps[i].Result = make(map[string]any, len(step.Result))
			for k, v := range step.Result {
				clone.Steps[i].Result[k] = v
			}
		}
	}

	// Deep copy saga data using JSON round-trip. This handles nested
	// maps and slices that a simple key-value copy would share by reference.
	if len(s.Data) > 0 {
		dataBytes, err := json.Marshal(s.Data)
		if err == nil {
			_ = json.Unmarshal(dataBytes, &clone.Data)
		}
	}

	return clone
}
