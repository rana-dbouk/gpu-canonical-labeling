# Accelerating Canonical Labeling for GPU-Parallel Graph Isomorphism

GPU-parallel canonical labeling for graph isomorphism based on the individualization--refinement framework underlying the state-of-the-art practical canonical-labeling tool [nauty](https://pallini.di.uniroma1.it/) [1].

The repository includes the proposed parallel GPU implementation together with an algorithmically-equivalent sequential version used as a baseline.

## Algorithm Overview

The GPU implementation exploits two levels of parallelism:

- **Tree parallelism:** different search-tree branches are explored by different CUDA thread blocks.
- **Node parallelism:** threads within a block cooperate on operations performed at a search-tree node, particularly partition refinement.

The search begins by distributing branches from the first branching level across thread blocks. Unlike `nauty`'s recursive sequential traversal, each block performs depth-first search using an explicit pre-allocated stack while maintaining its own mutable search state. The parallelization also incorporates dynamic workload balancing and mechanisms that preserve effective symmetry-based pruning across concurrently explored branches.

## Requirements

- CMake 3.24 or later
- C++17-compatible compiler
- NVIDIA CUDA Toolkit

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Usage

Compute the canonical labeling of one graph:

```bash
./build/gpu_canon_label <graph.txt>
```

The canonical adjacency list is written to `output`.

Check whether two graphs are isomorphic:

```bash
./build/gpu_canon_label <graph1.txt> <graph2.txt>
```

The program prints `isomorphic` or `not isomorphic`.

## Options

```text
--parallel       Use the parallel GPU implementation (default)
--sequential     Use the sequential implementation
--max-aut=N      Maximum number of stored automorphisms (default: 10000)
-h, --help       Show the help message
```

## Reference

[1] Brendan D. McKay, Adolfo Piperno,
Practical graph isomorphism, II,
Journal of Symbolic Computation,
Volume 60,
2014,
Pages 94-112,
ISSN 0747-7171,
https://doi.org/10.1016/j.jsc.2013.09.003.
(https://www.sciencedirect.com/science/article/pii/S0747717113001193)
