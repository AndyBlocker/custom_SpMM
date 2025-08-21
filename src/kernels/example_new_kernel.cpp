#include "kernel_interface.h"
#include "kernel_registry.h"
#include "MKL_Sparse_Methods.h"
#include <omp.h>
#include <cstring>

// Example of a new kernel implementation
// This demonstrates how to add a new SpMM kernel to the system
class ExampleNewKernel : public SpMMKernel {
public:
    std::string getName() const override {
        return "EXAMPLE_NEW_KERNEL";
    }
    
    std::string getDescription() const override {
        return "Example kernel showing how to add new SpMM implementations";
    }
    
    void execute(
        const sparse_matrix_t& A,
        const float* B,
        float* C,
        int M, int N, int K,
        const SpMMKernelConfig& config
    ) override {
        // Set number of threads from config
        omp_set_num_threads(config.num_threads);
        
        // Initialize output to zero
        std::memset(C, 0, M * K * sizeof(float));
        
        // Get CSR data from MKL sparse matrix
        sparse_index_base_t indexing;
        int rows, cols;
        int *rows_start, *rows_end, *col_indx;
        float *values;
        
        mkl_sparse_s_export_csr(A, &indexing, &rows, &cols, 
                               &rows_start, &rows_end, &col_indx, &values);
        
        // Simple parallel SpMM implementation
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < rows; i++) {
            for (int j = rows_start[i]; j < rows_end[i]; j++) {
                int col = col_indx[j];
                float val = values[j];
                
                // Multiply row of A with columns of B
                for (int k = 0; k < K; k++) {
                    C[i * K + k] += val * B[col * K + k];
                }
            }
        }
    }
    
    bool supportsConfig(const SpMMKernelConfig& config) const override {
        // This kernel supports all configurations
        return true;
    }
};

// Register the kernel automatically when this file is linked
REGISTER_KERNEL("EXAMPLE_NEW", ExampleNewKernel);