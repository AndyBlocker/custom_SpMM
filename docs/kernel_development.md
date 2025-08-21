# Kernel Development Guide

This guide explains how to add new SpMM kernels to the codebase after the reorganization.

## Directory Structure

```
src/
├── kernels/          # All SpMM kernel implementations
├── utils/            # Utility functions (CSR builder, tiling, etc.)
├── benchmarks/       # Benchmark applications
└── main.cpp          # Original main executable
```

## Adding a New Kernel

### Step 1: Create Your Kernel Class

Create a new file in `src/kernels/your_kernel_name.cpp`:

```cpp
#include "kernel_interface.h"
#include "kernel_registry.h"

class YourKernel : public SpMMKernel {
public:
    std::string getName() const override {
        return "YOUR_KERNEL_NAME";
    }
    
    std::string getDescription() const override {
        return "Description of your kernel";
    }
    
    void execute(
        const sparse_matrix_t& A,
        const float* B,
        float* C,
        int M, int N, int K,
        const SpMMKernelConfig& config
    ) override {
        // Your SpMM implementation here
    }
};

// Register the kernel
REGISTER_KERNEL("YOUR_KERNEL", YourKernel);
```

### Step 2: Update the Makefile

Add your kernel source to the `KERNEL_SRCS` in the Makefile:
```makefile
KERNEL_SRCS = $(wildcard $(KERNEL_DIR)/*.cpp)
```
This is automatic - no changes needed!

### Step 3: Build

```bash
make clean
make all
```

### Step 4: Use Your Kernel

Your kernel is automatically available in benchmarks:
```bash
./benchmark_unified --kernel YOUR_KERNEL --config configs/simple.yaml
```

## Kernel Configuration

The `SpMMKernelConfig` struct provides common configuration options:

```cpp
struct SpMMKernelConfig {
    int num_threads = 10;              // Number of OpenMP threads
    bool use_avx2 = true;             // Enable AVX2 instructions
    bool enable_prefetch = true;       // Enable prefetching
    int prefetch_distance = 4;         // Prefetch distance
    double cache_utilization_ratio = 0.5; // Cache utilization ratio
};
```

## Best Practices

1. **Memory Management**: Use the existing utility functions in `MKL_Sparse_Methods.h`
2. **Threading**: Respect the `config.num_threads` parameter
3. **Vectorization**: Check `config.use_avx2` before using AVX2 instructions
4. **Error Handling**: Validate inputs and handle edge cases
5. **Performance**: Consider cache optimization and memory access patterns

## Example Kernels

- `mkl_kernel.cpp`: Intel MKL sparse BLAS implementation
- `gustavson_kernel.cpp`: Custom Gustavson algorithm implementations
- `example_new_kernel.cpp`: Simple example showing the structure

## Testing Your Kernel

1. **Unit Testing**: Add tests in the `test/` directory
2. **Benchmarking**: Use the benchmark suite to compare performance
3. **Validation**: Ensure results match reference implementations

## Available Utilities

- `csr_builder.h/cpp`: Parallel CSR construction
- `tilling.h/cpp`: Compute optimal tiling parameters
- `make_lut.h/cpp`: Look-up table generation