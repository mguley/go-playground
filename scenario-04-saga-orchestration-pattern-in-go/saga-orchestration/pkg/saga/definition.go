package saga

import (
	"fmt"
)

// Definition describes a saga as a sequence of steps. It serves as
// a blueprint for creating saga instances and is typically defined
// once at startup and reused for multiple saga executions.
//
// The Definition is immutable after creation - modifying it while
// sagas are executing would cause undefined behavior.
type Definition struct {
	// Name identifies this saga definition (must be unique within the orchestrator)
	Name string

	// Description provides human-readable context about what this saga does
	Description string

	// Steps are the ordered sequence of steps that make up this saga.
	// Steps execute in order during forward execution and in reverse
	// order during compensation.
	Steps []*Step

	// stepIndex provides O(1) lookup of steps by ID
	stepIndex map[StepID]int
}

// NewDefinition creates a new saga definition with the given name.
// Steps should be added using AddStep().
func NewDefinition(name string) *Definition {
	return &Definition{
		Name:      name,
		Steps:     make([]*Step, 0),
		stepIndex: make(map[StepID]int),
	}
}

// WithDescription adds a description to the saga definition.
func (d *Definition) WithDescription(desc string) *Definition {
	d.Description = desc
	return d
}

// AddStep adds a step to the saga definition. Steps are executed
// in the order they are added during forward execution.
// Panics if a step with the same ID already exists - this is
// always a programming error caught at startup, not at runtime.
func (d *Definition) AddStep(step *Step) *Definition {
	if _, exists := d.stepIndex[step.ID]; exists {
		panic(fmt.Sprintf("duplicate step ID: %s", step.ID))
	}

	d.stepIndex[step.ID] = len(d.Steps)
	d.Steps = append(d.Steps, step)
	return d
}

// GetStep retrieves a step by its ID.
func (d *Definition) GetStep(id StepID) (*Step, bool) {
	index, ok := d.stepIndex[id]
	if !ok {
		return nil, false
	}
	return d.Steps[index], true
}

// GetStepByIndex retrieves a step by its position in the sequence.
func (d *Definition) GetStepByIndex(index int) (*Step, bool) {
	if index < 0 || index >= len(d.Steps) {
		return nil, false
	}
	return d.Steps[index], true
}

// StepCount returns the number of steps in this saga.
func (d *Definition) StepCount() int {
	return len(d.Steps)
}

// CreateInitialState creates the initial state for a new saga instance.
// Each step is initialized to PENDING status.
func (d *Definition) CreateInitialState(sagaID SagaID) *SagaState {
	steps := make([]StepState, len(d.Steps))
	for i, step := range d.Steps {
		steps[i] = StepState{
			ID:     step.ID,
			Name:   step.Name,
			Status: StepPending,
		}
	}
	return NewSagaState(sagaID, d.Name, steps)
}

// Validate checks that the saga definition is valid and ready for use.
func (d *Definition) Validate() error {
	if d.Name == "" {
		return fmt.Errorf("saga definition must have a name")
	}

	if len(d.Steps) == 0 {
		return fmt.Errorf("saga definition must have at least one step")
	}

	for i, step := range d.Steps {
		if step.ID == "" {
			return fmt.Errorf("step %d must have an ID", i)
		}
		if step.Name == "" {
			return fmt.Errorf("step %s must have a name", step.ID)
		}
		if step.Execute == nil {
			return fmt.Errorf("step %s must have an execute function", step.ID)
		}
	}

	return nil
}
