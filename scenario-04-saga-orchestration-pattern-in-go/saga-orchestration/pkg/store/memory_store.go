package store

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"sagaorchestration/pkg/saga"
)

// MemoryStore is an in-memory implementation of SagaStore.
// Suitable for development and testing; replace with a persistent
// store (PostgreSQL, Redis, etc.) for production use.
type MemoryStore struct {
	mu    sync.RWMutex
	sagas map[saga.SagaID]*saga.SagaState
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		sagas: make(map[saga.SagaID]*saga.SagaState),
	}
}

func (s *MemoryStore) Save(ctx context.Context, state *saga.SagaState) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Store a clone to prevent external modifications from affecting stored state
	s.sagas[state.ID] = state.Clone()
	return nil
}

func (s *MemoryStore) Get(ctx context.Context, id saga.SagaID) (*saga.SagaState, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	state, exists := s.sagas[id]
	if !exists {
		return nil, fmt.Errorf("saga %s not found", id)
	}

	return state.Clone(), nil
}

func (s *MemoryStore) List(ctx context.Context, filter SagaFilter) ([]*saga.SagaState, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	var results []*saga.SagaState

	for _, state := range s.sagas {
		if filter.Name != "" && state.Name != filter.Name {
			continue
		}
		if filter.Status != "" && state.Status != filter.Status {
			continue
		}
		if len(filter.Statuses) > 0 {
			found := false
			for _, status := range filter.Statuses {
				if state.Status == status {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		results = append(results, state.Clone())
	}

	// Sort by creation time (newest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].CreatedAt.After(results[j].CreatedAt)
	})

	// Apply pagination
	if filter.Offset > 0 {
		if filter.Offset >= len(results) {
			return []*saga.SagaState{}, nil
		}
		results = results[filter.Offset:]
	}

	if filter.Limit > 0 && len(results) > filter.Limit {
		results = results[:filter.Limit]
	}

	return results, nil
}

func (s *MemoryStore) Delete(ctx context.Context, id saga.SagaID) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.sagas, id)
	return nil
}

// GetStats returns statistics about the store contents (useful for monitoring).
func (s *MemoryStore) GetStats() map[string]int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := map[string]int{"total": len(s.sagas)}
	for _, state := range s.sagas {
		stats[string(state.Status)]++
	}
	return stats
}
