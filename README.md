### Go Playground

A hands-on learning environment for Go programming patterns, performance optimizations, and implementation strategies.
This repository contains practical scenarios that demonstrate Go concepts through guided exercises.

### Overview

This playground is designed to help you learn Go concepts by doing.
Each scenario focuses on a specific pattern or technique used in production Go environments.
The scenarios are self-contained and include step-by-step instructions, code samples, and explanations.

### Prerequisites

Before starting, ensure you have the following installed:
- [Go](https://golang.org/doc/install) (version 1.24+ recommended)
- A code editor of your choice
- Basic understanding of Go syntax and programming concepts

### Getting Started

```bash
# Clone this repository
git clone https://github.com/mguley/go-playground.git
cd go-playground

# View available scenarios
ls -la
```

### Available Scenarios

#### [Scenario 1: Escape Analysis in Go](./scenario-01-escape-analysis-in-go/)

Learn how the Go compiler optimizes memory allocation through escape analysis.
This scenario demonstrates how variables are allocated on the stack or heap, how to read escape analysis output, and
techniques to optimize performance by controlling memory allocation.

**Key Topics:**
- Understanding stack vs. heap allocation
- Reading compiler escape analysis output
- Benchmarking allocation performance
- Optimization techniques for memory efficiency

#### [Scenario 2: No Garbage Collection in Go](./scenario-02-no-gc-in-go/)

Explore the controversial technique of disabling Go's garbage collector for ultra-latency sensitive applications.
This scenario demonstrates the impact of garbage collection on performance, implements proper memory management strategies,
and explores hybrid approaches that balance predictable latency with memory stability.

**Key Topics:**
- Understanding garbage collection impact on performance
- Implementing object pooling and memory management strategies
- Benchmarking GC vs. no-GC scenarios
- Building hybrid approaches for production systems

#### [Scenario 3: Profiling in Go](./scenario-03-profiling-in-go/)

Master Go's built-in profiling tools to identify and fix performance bottlenecks in production applications.
This scenario builds a simulated content processing system and uses CPU, memory, goroutine, and trace profiling
to systematically diagnose and optimize performance issues.

**Key Topics:**
- CPU profiling to identify computational hot spots
- Memory profiling to detect leaks and allocation patterns
- Goroutine and block profiling for concurrency issues
- Execution trace analysis for deep runtime insights
- Continuous profiling infrastructure for production monitoring

#### [Scenario 4: Saga Orchestration pattern in Go](./scenario-04-saga-orchestration-pattern-in-go/)

Build resilient distributed transactions using the Saga Orchestration pattern.
This scenario implements a complete travel booking system that coordinates flight, hotel, and car rental
services, demonstrating how to handle failures gracefully through compensating transactions when operations
span multiple independent services.

**Key Topics:**
- Understanding distributed transactions and the limitations of traditional ACID
- Implementing the Saga Orchestration pattern for coordinating microservices
- Designing idempotent compensation functions for reliable rollback
- Building a flexible saga framework with retry and timeout handling
- Testing failure scenarios to validate compensation behavior
- Adding observability through events, logging, and metrics

#### [Scenario 5: Saga Choreography pattern in Go](./scenario-05-saga-choreography-pattern-in-go/)

Build distributed transactions using the Saga Choreography pattern, where services coordinate through events with no central orchestrator.
This scenario implements an e-commerce order processing system with four independent services that communicate exclusively through an event bus,
handling failures through automatic compensating transactions.

**Key Topics:**
- Understanding choreography-based saga coordination
- Designing event-driven communication between microservices
- Implementing compensating transactions for distributed rollback
- Building local caches to solve cross-service data availability
- State machine validation for concurrent event processing
- Idempotency guards for at-least-once event delivery
- Monitoring and observability for distributed workflows

#### [Scenario 6: Fuzz testing and property-based testing in Go](./scenario-06-fuzz-testing-in-go/)

Discover how Go's built-in fuzz testing framework uncovers defects that carefully crafted unit tests miss entirely.
This scenario builds a lightweight key-value data format from scratch - a text parser and a binary codec - and uses
coverage-guided fuzzing and property-based roundtrip testing to find crash-causing boundary errors, silent data
corruption, and subtle Go-specific pitfalls like UTF-8 replacement during rune iteration.

**Key Topics:**
- Understanding coverage-guided fuzz testing and how it differs from random input generation
- Writing crash-resistance fuzz tests to verify parsers never panic on arbitrary input
- Property-based roundtrip testing to ensure encode/decode consistency
- Interpreting fuzzer-discovered corpus files and tracing root causes
- Testing structural invariants across text serialization roundtrips
- Recognizing and avoiding silent data corruption from Go's rune iteration over non-UTF-8 strings

#### [Scenario 7: Event sourcing and CQRS in Go](./scenario-07-event-sourcing-and-cqrs-in-go/)

Build a complete bank account system using event sourcing and CQRS, where state is derived from an immutable
log of domain events rather than stored directly. This scenario implements an event store with optimistic
concurrency control, aggregates that enforce business rules through events, read-model projections for
efficient querying, and snapshots to keep event replay fast as streams grow.

**Key Topics:**
- Understanding event sourcing as an alternative to state-based persistence
- Building an append-only event store with optimistic concurrency control
- Implementing aggregates that derive state from event replay
- Separating reads from writes using CQRS projections
- Snapshotting long event streams for efficient aggregate loading
- Testing event-sourced systems with given-when-then style assertions
- Handling cross-aggregate operations and their consistency tradeoffs