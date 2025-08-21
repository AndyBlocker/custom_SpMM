# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a custom implementation of Sparse Matrix-Matrix Multiplication (SpMM) optimized for high performance using Intel MKL, OpenMP, and AVX2 instructions. The project implements multiple SpMM variants including single-threaded, multi-threaded, and Gustavson algorithm implementations.

## Build Commands

### Linux/Unix with GCC
```bash
g++ -O2 -march=native -fopenmp \
    main.cpp \
    MKL_Sparse_Methods.cpp \
    custom_spmm.cpp \
    custom_spmm_multi_thread.cpp \
    custom_spmm_single_thread.cpp \
    custom_spmm_yk.cpp \
    csr_builder.cpp \
    make_lut.cpp \
    tilling.cpp \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core \
    -lgomp -lpthread -lm -ldl \
    -o custom_spmm
```

### Windows with Visual Studio
Open `Project1.sln` in Visual Studio 2022 and build with x64 Release configuration.

### Run Command
```bash
./custom_spmm <sparsity> <M> <N> <K>
```
Example: `./custom_spmm 0.90 2048 2048 2048`

## Code Architecture

### Core Components

1. **MKL_Sparse_Methods.h/cpp**: Main interface for sparse matrix operations
   - Provides CSR (Compressed Sparse Row) construction from dense matrices
   - Implements multiple SpMM algorithms including MKL-based and custom Gustavson implementations
   - Key functions: `MKL_Sparse_CooXDense_Fast_Gustavson_new_yk()`

2. **custom_spmm_yk.cpp**: OpenMP-parallelized Gustavson SpMM implementation
   - Uses tiling for cache optimization
   - Implements AVX2 vectorization
   - Configurable tile sizes (Kc, Nb, Rb) based on L1/L2 cache sizes

3. **tilling.cpp**: Computes optimal tiling parameters
   - Balances between L1/L2 cache utilization
   - Adapts to matrix dimensions and available threads

4. **main.cpp**: Benchmarking driver
   - Runs warmup iterations (200) followed by test iterations (200)
   - Outputs timing results
   - Can write results to output.csv

### Key Design Patterns

1. **Memory Layout**: Row-major storage for all matrices
2. **Parallelization**: OpenMP with static/dynamic scheduling
3. **Vectorization**: AVX2 intrinsics for 8-float SIMD operations
4. **Cache Optimization**: Multi-level tiling targeting L1 (32KB) and L2 (1MB) caches

## Performance Considerations

- Default thread count is hardcoded to 10 in several OpenMP regions
- Tile sizes are dynamically computed based on cache utilization ratio (default 0.5)
- Memory alignment is critical - uses 64-byte alignment for accumulator tiles
- Prefetching is implemented with configurable distance (PREFETCH_P = 4)

## Current Branch: spmm_openmp

This branch focuses on OpenMP parallelization of the SpMM implementations.