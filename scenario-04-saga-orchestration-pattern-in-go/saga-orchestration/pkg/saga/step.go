package saga

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// StepFunc is the function signature for a saga step's forward action.
// It receives the saga context and state, and returns a result map and error.
// The result map can contain any data that subsequent steps might need.
type StepFunc func(ctx context.Context, state *SagaState) (map[string]any, error)

// CompensateFunc is the function signature for a step's compensation action.
// It receives the saga context and state, and should undo the work done by
// the corresponding forward action. Compensation functions must be
// idempotent - calling them multiple times has the same effect as calling once.
type CompensateFunc func(ctx context.Context, state *SagaState) error

// Step defines a single step in a saga. Each step has a forward action
// that performs the business logic and an optional compensation action
// that undoes the work if the saga needs to roll back.
type Step struct {
	// ID uniquely identifies this step within the saga
	ID StepID

	// Name is a human-readable name for the step, used in logging and UI
	Name string

	// Execute is the forward action that performs the step's work
	Execute StepFunc

	// Compensate is the action that undoes the step's work.
	// This is optional - some steps might not need compensation
	// (e.g., read-only steps or steps that are naturally idempotent)
	Compensate CompensateFunc

	// MaxRetries is the maximum number of times to retry this step
	// on transient failures before giving up
	MaxRetries int

	// RetryDelay is the initial delay between retry attempts.
	// The actual delay increases with each attempt (exponential backoff)
	RetryDelay time.Duration

	// Timeout is the maximum time allowed for this step to complete.
	// If zero, a default timeout will be used.
	Timeout time.Duration
}

// NewStep creates a new step with the given ID, name, and execute function.
// The step is created with sensible defaults: 3 retries, 100ms initial delay,
// and 30 second timeout. These can be customized using the builder methods.
func NewStep(id StepID, name string, execute StepFunc) *Step {
	return &Step{
		ID:         id,
		Name:       name,
		Execute:    execute,
		MaxRetries: 3,
		RetryDelay: 100 * time.Millisecond,
		Timeout:    30 * time.Second,
	}
}

// WithCompensation sets the compensation function for this step.
// Returns the step for method chaining.
func (s *Step) WithCompensation(compensate CompensateFunc) *Step {
	s.Compensate = compensate
	return s
}

// WithRetry configures retry behavior for this step.
// Returns the step for method chaining.
func (s *Step) WithRetry(maxRetries int, delay time.Duration) *Step {
	s.MaxRetries = maxRetries
	s.RetryDelay = delay
	return s
}

// WithTimeout sets the maximum execution time for this step.
// Returns the step for method chaining.
func (s *Step) WithTimeout(timeout time.Duration) *Step {
	s.Timeout = timeout
	return s
}

// ExecuteWithRetry executes the step with automatic retry on transient failures.
// It implements exponential backoff: each retry waits twice as long as the previous.
// Returns the result from a successful execution or the last error if all retries fail.
func (s *Step) ExecuteWithRetry(ctx context.Context, state *SagaState) (map[string]any, error) {
	var lastErr error

	for attempt := 0; attempt <= s.MaxRetries; attempt++ {
		// Check if context is cancelled before each attempt
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Create a timeout context for this specific attempt
		timeoutCtx, cancel := context.WithTimeout(ctx, s.Timeout)

		result, err := s.Execute(timeoutCtx, state)
		cancel() // Always cancel to release resources

		if err == nil {
			return result, nil
		}

		lastErr = err

		// Check if error is retryable - if not, fail immediately
		// and let the orchestrator begin compensation
		if !isRetryable(err) {
			return nil, err
		}

		// Don't sleep after the last attempt
		if attempt < s.MaxRetries {
			// Calculate backoff delay with exponential increase:
			// attempt 0 → delay×1, attempt 1 → delay×2, attempt 2 → delay×4
			delay := s.RetryDelay * time.Duration(1<<uint(attempt))

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
				// Continue to next attempt
			}
		}
	}

	return nil, fmt.Errorf("step %s failed after %d attempts: %w",
		s.Name, s.MaxRetries+1, lastErr)
}

// CompensateWithRetry executes the compensation with retry logic.
// Compensation uses more aggressive retrying than forward execution because
// leaving a step uncompensated creates an inconsistent state that may
// require manual intervention to resolve.
func (s *Step) CompensateWithRetry(ctx context.Context, state *SagaState) error {
	if s.Compensate == nil {
		return nil
	}

	var lastErr error

	// Ensure a minimum number of compensation attempts even if MaxRetries
	// is set low - compensation is too important to give up on quickly.
	maxAttempts := s.MaxRetries * 2
	if maxAttempts < 3 {
		maxAttempts = 3
	}

	for attempt := 0; attempt <= maxAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		timeoutCtx, cancel := context.WithTimeout(ctx, s.Timeout)
		err := s.Compensate(timeoutCtx, state)
		cancel()

		if err == nil {
			return nil
		}

		lastErr = err

		if attempt < maxAttempts {
			delay := s.RetryDelay * time.Duration(1<<uint(attempt))
			// Cap delay at 10 seconds for compensation to avoid excessive waits
			if delay > 10*time.Second {
				delay = 10 * time.Second
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}
	}

	return fmt.Errorf("compensation for step %s failed after %d attempts: %w",
		s.Name, maxAttempts+1, lastErr)
}

// retryableChecker is the interface that errors can implement to signal
// whether they represent transient failures worth retrying.
type retryableChecker interface {
	Retryable() bool
}

// isRetryable determines if an error is transient and worth retrying.
// It uses errors.As to walk the error chain, which correctly handles
// wrapped errors (e.g., fmt.Errorf("...: %w", serviceErr)).
func isRetryable(err error) bool {
	var checker retryableChecker
	if errors.As(err, &checker) {
		return checker.Retryable()
	}
	// By default, don't retry unknown errors - it's safer to fail fast
	// and begin compensation than to retry indefinitely
	return false
}

// RetryableError wraps an error to explicitly mark it as retryable.
// Use this when you want to signal a transient failure from code
// that doesn't use ServiceError.
type RetryableError struct {
	Err error
}

func (e *RetryableError) Error() string   { return e.Err.Error() }
func (e *RetryableError) Unwrap() error   { return e.Err }
func (e *RetryableError) Retryable() bool { return true }

// NewRetryableError creates a retryable error wrapper.
func NewRetryableError(err error) *RetryableError {
	return &RetryableError{Err: err}
}
