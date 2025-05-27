## Go Playground

A hands-on learning environment for Go programming patterns, performance optimizations, and implementation strategies.
This repository contains practical scenarios that demonstrate Go concepts through guided exercises.

## Overview

This playground is designed to help you learn Go concepts by doing.
Each scenario focuses on a specific pattern or technique used in production Go environments.
The scenarios are self-contained and include step-by-step instructions, code samples, and explanations.

## Prerequisites

Before starting, ensure you have the following installed:
- [Go](https://golang.org/doc/install) (version 1.24+ recommended)
- A code editor of your choice
- Basic understanding of Go syntax and programming concepts

## Getting Started

```bash
# Clone this repository
git clone https://github.com/mguley/go-playground.git
cd go-playground

# View available scenarios
ls -la
```

## Available Scenarios

### [Scenario 1: Escape Analysis in Go](./scenario-01-escape-analysis-in-go/)

Learn how the Go compiler optimizes memory allocation through escape analysis.
This scenario demonstrates how variables are allocated on the stack or heap, how to read escape analysis output, and
techniques to optimize performance by controlling memory allocation.

**Key Topics:**
- Understanding stack vs. heap allocation
- Reading compiler escape analysis output
- Benchmarking allocation performance
- Optimization techniques for memory efficiency

### [Scenario 2: No Garbage Collection in Go](./scenario-02-no-gc-in-go/)

Explore the controversial technique of disabling Go's garbage collector for ultra-latency sensitive applications.
This scenario demonstrates the impact of garbage collection on performance, implements proper memory management strategies,
and explores hybrid approaches that balance predictable latency with memory stability.

**Key Topics:**
- Understanding garbage collection impact on performance
- Implementing object pooling and memory management strategies
- Benchmarking GC vs. no-GC scenarios
- Building hybrid approaches for production systems

### [Scenario 3: Profiling in Go](./scenario-03-profiling-in-go/)

Master Go's built-in profiling tools to identify and fix performance bottlenecks in production applications.
This scenario builds a simulated content processing system and uses CPU, memory, goroutine, and trace profiling
to systematically diagnose and optimize performance issues.

**Key Topics:**
- CPU profiling to identify computational hot spots
- Memory profiling to detect leaks and allocation patterns
- Goroutine and block profiling for concurrency issues
- Execution trace analysis for deep runtime insights
- Continuous profiling infrastructure for production monitoring
