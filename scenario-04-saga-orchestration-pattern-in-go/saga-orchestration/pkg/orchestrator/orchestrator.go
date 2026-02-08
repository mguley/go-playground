package orchestrator

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"sagaorchestration/pkg/saga"
	"sagaorchestration/pkg/store"
)

// Orchestrator coordinates the execution of sagas. It manages the state
// machine transitions, handles step execution and compensation, and
// ensures durability through persistence.
type Orchestrator struct {
	definitions map[string]*saga.Definition
	store       store.SagaStore
	mu          sync.RWMutex

	// EventHandler is called when significant saga events occur.
	// Must be set before starting any sagas (not concurrency-safe
	// for writes after startup).
	EventHandler EventHandler

	// DefaultTimeout is the maximum time allowed for a complete saga
	DefaultTimeout time.Duration
}

// EventHandler is called for saga lifecycle events. Implementations
// can use this for logging, metrics collection, or alerting.
type EventHandler func(event SagaEvent)

// SagaEvent represents a significant event in saga execution.
type SagaEvent struct {
	Type      string
	SagaID    saga.SagaID
	SagaName  string
	StepID    saga.StepID
	StepName  string
	Status    string
	Error     string
	Timestamp time.Time
}

// NewOrchestrator creates a new orchestrator instance.
func NewOrchestrator(sagaStore store.SagaStore) *Orchestrator {
	return &Orchestrator{
		definitions:    make(map[string]*saga.Definition),
		store:          sagaStore,
		DefaultTimeout: 5 * time.Minute,
	}
}

// RegisterDefinition registers a saga definition with the orchestrator.
// Must be called before starting sagas of that type.
func (o *Orchestrator) RegisterDefinition(def *saga.Definition) error {
	if err := def.Validate(); err != nil {
		return fmt.Errorf("invalid saga definition: %w", err)
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, exists := o.definitions[def.Name]; exists {
		return fmt.Errorf("saga definition %s already registered", def.Name)
	}

	o.definitions[def.Name] = def
	log.Printf("Registered saga definition: %s with %d steps", def.Name, len(def.Steps))
	return nil
}

// GetDefinition retrieves a registered saga definition by name.
func (o *Orchestrator) GetDefinition(name string) (*saga.Definition, bool) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	def, ok := o.definitions[name]
	return def, ok
}

// StartSaga begins execution of a new saga instance. It creates the initial
// state, persists it, and begins executing asynchronously.
// Returns the saga ID which can be used to track progress.
func (o *Orchestrator) StartSaga(ctx context.Context, definitionName string,
	initialData map[string]any) (saga.SagaID, error) {

	def, ok := o.GetDefinition(definitionName)
	if !ok {
		return "", fmt.Errorf("saga definition %s not found", definitionName)
	}

	sagaID := saga.SagaID(fmt.Sprintf("%s-%d", definitionName, time.Now().UnixNano()))
	state := def.CreateInitialState(sagaID)

	// Populate initial data - typically the request that triggered the saga
	for k, v := range initialData {
		state.SetData(k, v)
	}

	// Persist initial state before starting execution
	if err := o.store.Save(ctx, state); err != nil {
		return "", fmt.Errorf("failed to save initial saga state: %w", err)
	}

	o.emitEvent(SagaEvent{
		Type:      "SAGA_STARTED",
		SagaID:    sagaID,
		SagaName:  def.Name,
		Status:    string(saga.StatusPending),
		Timestamp: time.Now(),
	})

	// Execute in a goroutine - the caller polls for status via GetSagaState.
	// We use context.Background() so the saga outlives the HTTP request.
	go o.executeSaga(context.Background(), def, state)

	return sagaID, nil
}

// executeSaga is the main orchestration loop. It handles forward execution,
// failure detection, and compensation.
func (o *Orchestrator) executeSaga(ctx context.Context, def *saga.Definition, state *saga.SagaState) {
	ctx, cancel := context.WithTimeout(ctx, o.DefaultTimeout)
	defer cancel()

	// Transition to running
	state.Status = saga.StatusRunning
	state.UpdatedAt = time.Now()
	if err := o.store.Save(ctx, state); err != nil {
		log.Printf("Failed to save saga state: %v", err)
	}

	o.emitEvent(SagaEvent{
		Type: "SAGA_RUNNING", SagaID: state.ID, SagaName: state.Name,
		Status: string(saga.StatusRunning), Timestamp: time.Now(),
	})

	// === Forward execution ===
	var failedStep int = -1
	var executionError error

	for i := state.CurrentStep; i < len(def.Steps); i++ {
		step := def.Steps[i]
		stepState := &state.Steps[i]

		now := time.Now()
		stepState.StartedAt = &now
		stepState.Status = saga.StepRunning

		result, err := step.ExecuteWithRetry(ctx, state)

		if err != nil {
			failedStep = i
			executionError = err
			stepState.Status = saga.StepFailed
			stepState.Error = err.Error()

			o.emitEvent(SagaEvent{
				Type: "STEP_FAILED", SagaID: state.ID, SagaName: state.Name,
				StepID: step.ID, StepName: step.Name,
				Status: string(saga.StepFailed), Error: err.Error(),
				Timestamp: time.Now(),
			})
			break
		}

		// Step succeeded
		stepState.Status = saga.StepCompleted
		stepState.Result = result
		completedAt := time.Now()
		stepState.CompletedAt = &completedAt
		state.CurrentStep = i + 1

		// Store step results in saga data for subsequent steps
		if result != nil {
			for k, v := range result {
				state.SetData(k, v)
			}
		}

		o.emitEvent(SagaEvent{
			Type: "STEP_COMPLETED", SagaID: state.ID, SagaName: state.Name,
			StepID: step.ID, StepName: step.Name,
			Status: string(saga.StepCompleted), Timestamp: time.Now(),
		})

		// Persist after each step for durability
		if err = o.store.Save(ctx, state); err != nil {
			log.Printf("Failed to save saga state: %v", err)
		}
	}

	// === Check for success ===
	if failedStep == -1 {
		state.Status = saga.StatusCompleted
		now := time.Now()
		state.CompletedAt = &now
		state.UpdatedAt = now

		o.emitEvent(SagaEvent{
			Type: "SAGA_COMPLETED", SagaID: state.ID, SagaName: state.Name,
			Status: string(saga.StatusCompleted), Timestamp: time.Now(),
		})

		if err := o.store.Save(ctx, state); err != nil {
			log.Printf("Failed to save saga state: %v", err)
		}
		return
	}

	// === Compensation ===
	state.Status = saga.StatusCompensating
	state.Error = executionError.Error()
	state.UpdatedAt = time.Now()

	o.emitEvent(SagaEvent{
		Type: "SAGA_COMPENSATING", SagaID: state.ID, SagaName: state.Name,
		Status: string(saga.StatusCompensating), Error: executionError.Error(),
		Timestamp: time.Now(),
	})

	if err := o.store.Save(ctx, state); err != nil {
		log.Printf("Failed to save saga state: %v", err)
	}

	// Compensate in reverse order, starting from the step BEFORE the failed one.
	// The failed step never completed its work, so it doesn't need compensation.
	compensationFailed := false
	for i := failedStep - 1; i >= 0; i-- {
		step := def.Steps[i]
		stepState := &state.Steps[i]

		// Only compensate steps that actually completed
		if stepState.Status != saga.StepCompleted {
			stepState.Status = saga.StepSkipped
			continue
		}

		if step.Compensate == nil {
			// No compensation defined - consider it handled
			continue
		}

		stepState.Status = saga.StepCompensating
		err := step.CompensateWithRetry(ctx, state)

		if err != nil {
			stepState.Status = saga.StepCompensationFailed
			stepState.Error = err.Error()
			compensationFailed = true

			o.emitEvent(SagaEvent{
				Type: "STEP_COMPENSATION_FAILED", SagaID: state.ID,
				SagaName: state.Name, StepID: step.ID, StepName: step.Name,
				Status: string(saga.StepCompensationFailed), Error: err.Error(),
				Timestamp: time.Now(),
			})

			// Important: continue compensating remaining steps even if one fails.
			// We want to undo as much as possible rather than leaving everything
			// in an inconsistent state.
		} else {
			stepState.Status = saga.StepCompensated
			compensatedAt := time.Now()
			stepState.CompensatedAt = &compensatedAt

			o.emitEvent(SagaEvent{
				Type: "STEP_COMPENSATED", SagaID: state.ID, SagaName: state.Name,
				StepID: step.ID, StepName: step.Name,
				Status: string(saga.StepCompensated), Timestamp: time.Now(),
			})
		}

		if err = o.store.Save(ctx, state); err != nil {
			log.Printf("Failed to save saga state: %v", err)
		}
	}

	// Set final status
	if compensationFailed {
		state.Status = saga.StatusFailed
		o.emitEvent(SagaEvent{
			Type: "SAGA_FAILED", SagaID: state.ID, SagaName: state.Name,
			Status: string(saga.StatusFailed), Timestamp: time.Now(),
		})
	} else {
		state.Status = saga.StatusCompensated
		o.emitEvent(SagaEvent{
			Type: "SAGA_COMPENSATED", SagaID: state.ID, SagaName: state.Name,
			Status: string(saga.StatusCompensated), Timestamp: time.Now(),
		})
	}

	now := time.Now()
	state.CompletedAt = &now
	state.UpdatedAt = now

	if err := o.store.Save(ctx, state); err != nil {
		log.Printf("Failed to save saga state: %v", err)
	}
}

// GetSagaState retrieves the current state of a saga by ID.
func (o *Orchestrator) GetSagaState(ctx context.Context, sagaID saga.SagaID) (*saga.SagaState, error) {
	return o.store.Get(ctx, sagaID)
}

// ListSagas retrieves sagas matching the given criteria.
func (o *Orchestrator) ListSagas(ctx context.Context, filter store.SagaFilter) ([]*saga.SagaState, error) {
	return o.store.List(ctx, filter)
}

func (o *Orchestrator) emitEvent(event SagaEvent) {
	if o.EventHandler != nil {
		o.EventHandler(event)
	}
}
