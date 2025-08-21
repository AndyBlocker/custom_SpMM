#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <string>
#include <memory>
#include <mkl_spblas.h>

struct SpMMKernelConfig {
    int num_threads = 10;
    bool use_avx2 = true;
    bool enable_prefetch = true;
    int prefetch_distance = 4;
    double cache_utilization_ratio = 0.5;
};

class SpMMKernel {
public:
    virtual ~SpMMKernel() = default;
    
    virtual std::string getName() const = 0;
    
    virtual std::string getDescription() const = 0;
    
    virtual void execute(
        const sparse_matrix_t& A,
        const float* B,
        float* C,
        int M, int N, int K,
        const SpMMKernelConfig& config = SpMMKernelConfig()
    ) = 0;
    
    virtual bool supportsConfig(const SpMMKernelConfig& config) const {
        return true;
    }
};

using SpMMKernelPtr = std::unique_ptr<SpMMKernel>;

#endif // KERNEL_INTERFACE_H