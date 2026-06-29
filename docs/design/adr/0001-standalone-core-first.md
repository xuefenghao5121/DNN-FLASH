# ADR-0001: Build standalone attention core before TensorFlow/XLA integration

## Status

Accepted

## Context

FlashOne has several high-risk integration points: XLA pattern preservation, oneDNN post-op expressiveness, online softmax lowering, and hardware-specific kernel performance. Starting directly inside TensorFlow/XLA would make failures hard to isolate.

## Decision

Build a standalone C++17 + Python/Numpy reference core first. The core must prove the online softmax recurrence, masking semantics, correctness tolerances, and benchmark harness before replacing inner loops with oneDNN or registering TensorFlow/XLA entry points.

## Consequences

- Faster debugging and deterministic correctness tests.
- Initial code is not yet the final high-performance backend.
- oneDNN/TensorFlow integration becomes an incremental backend replacement instead of the first unknown.
