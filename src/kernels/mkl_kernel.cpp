#include "kernel_interface.h"
#include "kernel_registry.h"
#include "MKL_Sparse_Methods.h"
#include <mkl.h>

class MKLKernel : public SpMMKernel {
public:
    std::string getName() const override {
        return "MKL_SPARSE";
    }
    
    std::string getDescription() const override {
        return "Intel MKL Sparse BLAS implementation of SpMM";
    }
    
    void execute(
        const sparse_matrix_t& A,
        const float* B,
        float* C,
        int M, int N, int K,
        const SpMMKernelConfig& config
    ) override {
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        mkl_sparse_s_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            1.0f,
            A,
            descr,
            SPARSE_LAYOUT_ROW_MAJOR,
            B,
            K,
            K,
            0.0f,
            C,
            K
        );
    }
};

REGISTER_KERNEL("MKL_SPARSE", MKLKernel);