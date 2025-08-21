# SpMM Benchmark Scripts

This directory contains the organized build and run scripts for the SpMM benchmark suite.

## Quick Start

### Simple Benchmark (4 specific test cases)
```bash
./run_benchmark.sh --preset simple
```

### Build Only
```bash
./build.sh --target unified
```

### Run with Custom Settings
```bash
./run_benchmark.sh --preset extended --warmup 20 --test 100 --threads 8
```

## Scripts Overview

### `build.sh`
Master build script for all benchmark variants.

**Usage:**
```bash
./build.sh [OPTIONS]
  --debug             Build with debug symbols
  --target <name>     Build specific target (unified, simple, extended, original, all)
  --clean             Clean build directory
  --help              Show help message
```

**Examples:**
```bash
# Build all targets
./build.sh

# Build only the unified benchmark
./build.sh --target unified

# Clean and rebuild with debug symbols
./build.sh --clean
./build.sh --debug
```

### `run_benchmark.sh`
Unified benchmark runner with preset configurations.

**Usage:**
```bash
./run_benchmark.sh [OPTIONS]
  --preset, -p <name>   Benchmark preset (see below)
  --output, -o <file>   Output CSV file
  --warmup <n>          Number of warmup iterations
  --test <n>            Number of test iterations
  --threads <n>         Number of threads to use
  --quiet, -q           Suppress verbose output
  --build               Build the benchmark before running
  --help, -h            Show help message
```

**Available Presets:**
- `simple` - Your 4 specific test cases (197×3072×768, etc.)
- `square` - Square matrices (512² to 4096²)
- `tall-skinny` - Tall-skinny matrices
- `extended` - Full extended benchmark suite
- `quick` - Quick test for debugging

**Examples:**
```bash
# Run simple benchmark quietly
./run_benchmark.sh --preset simple --quiet

# Run extended benchmark with custom output
./run_benchmark.sh --preset extended --output my_results.csv

# Quick test with reduced iterations
./run_benchmark.sh --preset quick --warmup 2 --test 5
```

## Output Files

All benchmark results are saved to the `results/` directory by default:
- Format: `benchmark_<preset>_<timestamp>.csv`
- Example: `benchmark_simple_20240113_142530.csv`

## Directory Structure

```
custom_SpMM/
├── scripts/           # All scripts
│   ├── build.sh       # Master build script
│   ├── run_benchmark.sh # Unified runner
│   └── README.md      # This file
├── src/               # Source code
├── include/           # Header files
├── build/             # Build artifacts (gitignored)
└── results/           # Benchmark results (gitignored)
```

## Environment Setup

The scripts automatically detect MKL installation. If MKL is not found, you can:

1. Source Intel oneAPI environment:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

2. Or set MKLROOT manually:
   ```bash
   export MKLROOT=/path/to/mkl
   ```

## Troubleshooting

### MKL Libraries Not Found
If you get library errors at runtime, the scripts will automatically set up the library path. If issues persist:
```bash
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
```

### Build Errors
Check that you have:
- GCC with OpenMP support
- Intel MKL installed
- Required permissions in the directory

### Performance Issues
- The scripts automatically reduce iterations for very large matrices
- Use `--threads` to control parallelism
- Use `--quiet` to reduce output overhead during benchmarking