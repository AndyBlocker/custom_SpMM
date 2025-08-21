#include "kernel_interface.h"
#include "kernel_registry.h"
#include "MKL_Sparse_Methods.h"
#include <omp.h>
#include <mkl.h>
#include <vector>
#include <stdexcept>

// Note: The Gustavson kernels in this codebase expect dense matrix A as input
// and perform the sparse conversion internally. This is different from the
// SpMMKernel interface which expects a sparse matrix. We need to convert back.

class GustavsonKernel : public SpMMKernel {
private:
    // Helper to convert sparse matrix back to dense for these kernels
    std::vector<float> sparseToDense(const sparse_matrix_t& A, int M, int N) const {
        std::vector<float> dense(M * N, 0.0f);
        
        // Export CSR data from MKL sparse matrix
        sparse_index_base_t indexing;
        int rows, cols;
        int *rows_start, *rows_end, *col_indx;
        float *values;
        
        mkl_sparse_s_export_csr(const_cast<sparse_matrix_t&>(A), &indexing, 
                               &rows, &cols, &rows_start, &rows_end, &col_indx, &values);
        
        // Convert to dense
        for (int i = 0; i < rows; i++) {
            for (int j = rows_start[i]; j < rows_end[i]; j++) {
                int col = col_indx[j];
                dense[i * N + col] = values[j];
            }
        }
        
        return dense;
    }
    
public:
    std::string getName() const override {
        return "GUSTAVSON";
    }
    
    std::string getDescription() const override {
        return "Custom Gustavson algorithm implementation with OpenMP";
    }
    
    void execute(
        const sparse_matrix_t& A,
        const float* B,
        float* C,
        int M, int N, int K,
        const SpMMKernelConfig& config
    ) override {
        // This implementation expects dense matrix A, so we need to convert
        // Note: This is inefficient but maintains compatibility with the existing code
        auto denseA = sparseToDense(A, M, N);
        
        // Convert to 2D array format expected by the function
        std::vector<float*> A_rows(M);
        std::vector<float*> B_rows(N);
        std::vector<float*> C_rows(M);
        
        for (int i = 0; i < M; i++) {
            A_rows[i] = denseA.data() + i * N;
            C_rows[i] = C + i * K;
        }
        for (int i = 0; i < N; i++) {
            B_rows[i] = const_cast<float*>(B) + i * K;
        }
        
        omp_set_num_threads(config.num_threads);
        MKL_Sparse_CooXDense_Fast_Gustavson(A_rows.data(), B_rows.data(), 
                                           C_rows.data(), M, N, K, 0);
    }
};

class GustavsonNewKernel : public SpMMKernel {
private:
    std::vector<float> sparseToDense(const sparse_matrix_t& A, int M, int N) const {
        std::vector<float> dense(M * N, 0.0f);
        
        sparse_index_base_t indexing;
        int rows, cols;
        int *rows_start, *rows_end, *col_indx;
        float *values;
        
        mkl_sparse_s_export_csr(const_cast<sparse_matrix_t&>(A), &indexing, 
                               &rows, &cols, &rows_start, &rows_end, &col_indx, &values);
        
        for (int i = 0; i < rows; i++) {
            for (int j = rows_start[i]; j < rows_end[i]; j++) {
                int col = col_indx[j];
                dense[i * N + col] = values[j];
            }
        }
        
        return dense;
    }
    
public:
    std::string getName() const override {
        return "GUSTAVSON_NEW_YK";
    }
    
    std::string getDescription() const override {
        return "Optimized Gustavson with AVX2 vectorization and cache tiling";
    }
    
    void execute(
        const sparse_matrix_t& A,
        const float* B,
        float* C,
        int M, int N, int K,
        const SpMMKernelConfig& config
    ) override {
        // Convert sparse to dense (inefficient but maintains compatibility)
        auto denseA = sparseToDense(A, M, N);
        
        omp_set_num_threads(config.num_threads);
        MKL_Sparse_CooXDense_Fast_Gustavson_new_yk(denseA.data(), 
                                                   const_cast<float*>(B), C, M, N, K, 0);
    }
    
    bool supportsConfig(const SpMMKernelConfig& config) const override {
        return config.use_avx2;
    }
};

REGISTER_KERNEL("GUSTAVSON", GustavsonKernel);
REGISTER_KERNEL("GUSTAVSON_NEW_YK", GustavsonNewKernel);