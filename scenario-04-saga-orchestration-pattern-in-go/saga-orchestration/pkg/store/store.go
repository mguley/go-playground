package store

import (
	"context"

	"sagaorchestration/pkg/saga"
)

// SagaStore defines the interface for saga state persistence.
// Implementations can use various backends: in-memory for development,
// PostgreSQL for production, Redis for distributed systems, etc.
type SagaStore interface {
	// Save persists the saga state (upsert: create if not exists, update if exists)
	Save(ctx context.Context, state *saga.SagaState) error

	// Get retrieves a saga state by ID
	Get(ctx context.Context, id saga.SagaID) (*saga.SagaState, error)

	// List retrieves sagas matching the given filter criteria
	List(ctx context.Context, filter SagaFilter) ([]*saga.SagaState, error)

	// Delete removes a saga state (used for cleanup of old completed sagas)
	Delete(ctx context.Context, id saga.SagaID) error
}

// SagaFilter defines criteria for listing sagas. All fields are optional;
// empty/zero fields mean "no filter on this field".
type SagaFilter struct {
	Name     string
	Status   saga.SagaStatus
	Statuses []saga.SagaStatus
	Limit    int
	Offset   int
}
