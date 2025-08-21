/*  Demo：验证优化后的 Sparse × Dense
 *  说明：
 *    • 默认矩阵规模 2048 × 2048，sparsity=0.90 表示 90 % 元素为 0
 *    • 可通过命令行参数修改稀疏度：  ./demo 0.85
 *    • 代码同时给出等尺寸 Dense×Dense 基准
 */

#include "../include/MKL_Sparse_Methods.h"
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#define _CRT_SECURE_NO_WARNINGS

static double SparseDenseGustavsonDemo(int M, int N, int K,
    int flag,      /* 0:A·B  1:Aᵀ·B  2:Aᴴ·B */
    float sparsity);       /* 0.0‑1.0 */

static double SparseDenseGustavsonDemoNew(int M, int N, int K,
    int flag,      /* 0:A·B  1:Aᵀ·B  2:Aᴴ·B */
    float sparsity);       /* 0.0‑1.0 */

static double SparseDenseDemo(int M, int N, int K,
    int flag,      /* 0:A·B  1:Aᵀ·B  2:Aᴴ·B */
    float sparsity);       /* 0.0‑1.0 */

static double DenseDenseDemo(int M, int N, int K);


FILE* safe_fopen(const char* filename, const char* mode) {
#ifdef _WIN32
    FILE* fp = nullptr;
    if (fopen_s(&fp, filename, mode) != 0) {
        return nullptr;
    }
    return fp;
#else
    return fopen(filename, mode);
#endif
}

void write_csv_row(int M, int N, int K, double sparsity,
    double t1, double t2, double t3, int test_iter) {
    FILE* F = safe_fopen("output.csv", "ab");
    if (!F) {
        perror("Failed to open output.csv");
        exit(EXIT_FAILURE);
    }

    fprintf(F, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
        M, N, K, sparsity,
        t1 / test_iter, t2 / test_iter, t3 / test_iter);

    fclose(F);
}


int main(int argc, char** argv)
{
    int warmup_iter = 200;
    int test_iter = 200;
    const int M = static_cast<int>(atof(argv[2])), N = static_cast<int>(atof(argv[3])), K = static_cast<int>(atof(argv[4]));
    float sparsity = 0.90f;
    if (argc > 1) sparsity = static_cast<float>(atof(argv[1]));
    printf("M=%d,N=%d,K=%d,sparsity=%.6f\n", M, N, K, sparsity);

    //for (int i = 0; i < warmup_iter; i++) {
    //    // if(i == 0)
    //    //     printf("\n===== WramUp Sparse × Dense  (sparsity = %.2f) =====\n", sparsity);
    //    // else
    //    //     printf("===== WramUp Sparse × Dense  (sparsity = %.2f) =====\n", sparsity);
    //    SparseDenseDemo(M, N, K, 0, sparsity);
    //}

    //// printf("\033[0m\033[43;31;1;5m=====  Sparse × Dense  (sparsity = %.2f) =====\033[0m\n", sparsity);
    //double spend_time1 = 0.0;
    //for (int i = 0; i < test_iter; i++) {
    //    spend_time1 = spend_time1 + SparseDenseDemo(M, N, K, 0, sparsity);
    //}
    //printf("Sparse×Dense (fast): %.6f s\n", spend_time1 / test_iter);

    for (int i = 0; i < warmup_iter; i++) {
        // if(i == 0)
        //     printf("\n===== WramUp Gustavson Sparse × Dense  (sparsity = %.2f) =====\n", sparsity);
        // else
        //     printf("===== WramUp Gustavson Sparse × Dense  (sparsity = %.2f) =====\n", sparsity);
        SparseDenseGustavsonDemoNew(M, N, K, 0, sparsity);
    }

    // printf("\033[0m\033[43;31;1;5m=====  Gustavson Sparse × Dense  (sparsity = %.2f) =====\033[0m\n", sparsity);
    double spend_time2 = 0.0;
    for (int i = 0; i < test_iter; i++) {
        spend_time2 = spend_time2 + SparseDenseGustavsonDemoNew(M, N, K, 0, sparsity);
    }
    printf("Sparse×Dense (Gustavson New): %.6f s\n", spend_time2 / test_iter);

    //for (int i=0;i<warmup_iter;i++){
    //     // if(i == 0)
    //     //     printf("\n===== WramUp Dense  × Dense  (MKL SGEMM)  =====\n");
    //     // else
    //     //     printf("===== WramUp Dense  × Dense  (MKL SGEMM)  =====\n");
    //     DenseDenseDemo(M, N, K);
    //}

    //// printf("\033[0m\033[43;31;1;5m=====  Dense  × Dense  (MKL SGEMM)  =====\033[0m\n");
    //double spend_time3 = 0.0;
    //for (int i=0;i<test_iter;i++){
    //    spend_time3 = spend_time3 + DenseDenseDemo(M, N, K);
    //}
    //printf("Dense×Dense (mkl gemm): %.6f s\n", spend_time3/test_iter);

    //write_csv_row(M, N, K, sparsity, spend_time1, spend_time2, spend_time3, test_iter);
    return 0;
}

static void print_matrix(float** A, int M, int N) {
    printf("=======================%d x %d======================\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

/*------------------------------------------------------------
 *  稀疏 × 稠密 demo
 *-----------------------------------------------------------*/
static double SparseDenseDemo(int M, int N, int K,
    int flag, float sparsity)
{
    /* 1. 生成稀疏 A（Row‑Major）、稠密 B */
    float** A = alloc2float(N, M);           /* M×N */
    float** B = alloc2float(K, N);           /* N×K */
    float** C = alloc2float(K, M);           /* M×K */

    std::srand((unsigned)std::time(nullptr));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float r = static_cast<float>(std::rand()) / RAND_MAX;
            if (r > sparsity) {
                A[i][j] = static_cast<float>(std::rand() % 11 - 5);
            }
            else A[i][j] = 0.0f;
        }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            B[i][j] = static_cast<float>(std::rand() % 11 - 5);

    auto t_start = std::chrono::steady_clock::now();
    /* 2. 计时并调用优化版乘法 */
    //for (int i = 0; i < 2000; i++) {
    MKL_Sparse_CooXDense_Fast(A,
        B, C,
        M, N, K, flag);
    //}
    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t_end - t_start;
    // printf("Sparse×Dense (fast): %.6f s\n",
    //        diff.count());

    /* 3. 清理 */
    // print_matrix(A, M, N);
    // print_matrix(B, N, K);
    // print_matrix(C, M, K);
    free2float(A); free2float(B); free2float(C);
    return diff.count();
}


bool verify_result(float** denseA, float** denseB, float** denseC_gustavson,
    int rowsA, int colsA, int colsC, float tolerance = 1e-5) {
    // 分配存储正确结果的矩阵
    std::vector<float> correct_result_data(rowsA * colsC, 0.0f);
    float* correct_result = correct_result_data.data();

    // 传统三重循环计算正确结果
    auto start_verify = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsC; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += denseA[0][i * colsA + k] * denseB[0][k * colsC + j];
            }
            correct_result[i * colsC + j] = sum;
        }
    }

    auto end_verify = std::chrono::high_resolution_clock::now();
    auto verify_time = std::chrono::duration_cast<std::chrono::microseconds>(end_verify - start_verify);

    printf("Verification computation time: %ld μs\n", verify_time.count());

    // 对比结果
    bool is_correct = true;
    int error_count = 0;
    const int max_errors_to_show = 10;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsC; ++j) {
            float diff = std::abs(correct_result[i * colsC + j] - denseC_gustavson[0][i * colsC + j]);
            if (diff > tolerance) {
                is_correct = false;
                if (error_count < max_errors_to_show) {
                    printf("ERROR at [%d][%d]: correct=%.6f, gustavson=%.6f, diff=%.6f\n",
                        i, j, correct_result[i * colsC + j], denseC_gustavson[0][i * colsC + j], diff);
                }
                error_count++;
            }
        }
    }

    // 输出结果
    if (is_correct) {
        // 输出绿色的"correct"
        printf("\033[1;32mcorrect\033[0m\n");
        printf("All %d elements matched! ✓\n", rowsA * colsC);
    }
    else {
        printf("\033[1;31mINCORRECT\033[0m\n");
        printf("Found %d errors out of %d elements\n", error_count, rowsA * colsC);
    }

    return is_correct;
}


bool verify_result_1d(const float* denseA, const float* denseB, const float* denseC_gustavson,
    int rowsA, int colsA, int colsC, float tolerance = 1e-5f) {
    // 分配存储正确结果的扁平化矩阵
    std::vector<float> correct_result(rowsA * colsC, 0.0f);
    float* correct = correct_result.data();

    // 传统三重循环计算正确结果
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsC; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                // A[i][k] -> denseA[i*colsA + k]
                // B[k][j] -> denseB[k*colsC + j]
                sum += denseA[i * colsA + k] * denseB[k * colsC + j];
            }
            correct[i * colsC + j] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Verification time: %ld μs\n", us.count());

    // 对比结果
    bool ok = true;
    int errors = 0;
    const int max_show = 10;
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsC; ++j) {
            float ref = correct[i * colsC + j];
            float got = denseC_gustavson[i * colsC + j];
            float diff = std::fabs(ref - got);
            if (diff > tolerance) {
                ok = false;
                if (errors < max_show) {
                    printf("ERROR [%d,%d]: ref=%.6f, gust=%.6f, diff=%.6f\n",
                        i, j, ref, got, diff);
                }
                ++errors;
            }
        }
    }

    if (ok) {
        printf("\033[1;32mcorrect\033[0m: all %d elements matched ✓\n", rowsA * colsC);
    }
    else {
        printf("\033[1;31mINCORRECT\033[0m: %d errors out of %d elements\n",
            errors, rowsA * colsC);
    }
    return ok;
}

/*------------------------------------------------------------
 *  稀疏 × 稠密 demo
 *-----------------------------------------------------------*/
static double SparseDenseGustavsonDemo(int M, int N, int K,
    int flag, float sparsity)
{
    /* 1. 生成稀疏 A（Row‑Major）、稠密 B */
    float** A = alloc2float(N, M);           /* M×N */
    float** B = alloc2float(K, N);           /* N×K */
    float** C = alloc2float(K, M);           /* M×K */

    std::srand(1024);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float r = static_cast<float>(std::rand()) / RAND_MAX;
            if (r > sparsity) {
                A[i][j] = static_cast<float>(std::rand() % 11 - 5);
            }
            else A[i][j] = 0.0f;
        }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            B[i][j] = static_cast<float>(std::rand() % 11 - 5);

    auto t_start = std::chrono::steady_clock::now();

    /* 3. 计时并调用优化版乘法 */
    bool success = MKL_Sparse_CooXDense_Fast_Gustavson(A,
        B, C,
        M, N, K, flag);
    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t_end - t_start;

    // if (success) {            
    //         // 进行对拍验证（只在矩阵不太大时进行，避免验证时间过长）
    //         if (M * K <= 1000000) { // 100万元素以内
    //             printf("Starting verification...\n");
    //             verify_result(A, B, C, M, N, K);
    //         } else {
    //             printf("Matrix too large for verification (%d elements), skipping...\n", M * K);
    //         }
    //     } else {
    //         printf("Gustavson SpMM failed!\n");
    //     }

    // printf("Sparse×Dense (fast): %.6f s\n",
    //        diff.count());

    /* 4. 清理 */
    // print_matrix(A, M, N);
    // print_matrix(B, N, K);
    // print_matrix(C, M, K);
    free2float(A); free2float(B); free2float(C);
    return diff.count();
}

#include <random>

static double SparseDenseGustavsonDemoNew(int M, int N, int K,
    int flag, float sparsity)
{
    /* 1. 生成稀疏 A（Row-Major）、稠密 B */
    float* A = (float*)std::malloc(sizeof(float) * M * N);           /* M×N */
    float* B = (float*)std::malloc(sizeof(float) * N * K);           /* N×K */
    float* C = (float*)std::malloc(sizeof(float) * M * K);           /* M×K */
    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        std::exit(1);
    }

    // 固定随机数生成器（Mersenne Twister）
    std::srand(1024);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float r = static_cast<float>(std::rand()) / RAND_MAX;
            if (r > sparsity) {
                A[i * N + j] = static_cast<float>(std::rand() % 11 - 5);
            }
            else {
                A[i * N + j] = 0.0f;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            B[i * K + j] = static_cast<float>(std::rand() % 11 - 5);
        }
    }

    auto t_start = std::chrono::steady_clock::now();

    bool success = false;
    //for (int i = 0; i < 2000; i++) {
    success = MKL_Sparse_CooXDense_Fast_Gustavson_new_yk(
        A, B, C, M, N, K, flag
    );
    //}

    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t_end - t_start;

    //if (success) {
    //    if (M * K <= 1000000) { // 验证
    //        printf("Starting verification...\n");
    //        verify_result_1d(A, B, C, M, N, K);
    //    }
    //    else {
    //        printf("Matrix too large for verification (%d elements), skipping...\n", M * K);
    //    }
    //}
    //else {
    //    printf("Gustavson SpMM failed!\n");
    //}

    if (A) { std::free(A); A = nullptr; }
    if (B) { std::free(B); B = nullptr; }
    if (C) { std::free(C); C = nullptr; }
    return diff.count();
}




/*------------------------------------------------------------
 *  稠密 × 稠密 baseline
 *-----------------------------------------------------------*/
static double DenseDenseDemo(int M, int N, int K)
{
    float** A = alloc2float(N, M);       /* M×N */
    float** B = alloc2float(K, N);       /* N×K */
    float** C = alloc2float(K, M);       /* M×K */

    std::srand((unsigned)std::time(nullptr));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            A[i][j] = static_cast<float>(std::rand() % 11 - 5);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            B[i][j] = static_cast<float>(std::rand() % 11 - 5);

    auto t_start = std::chrono::steady_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, K, N,
        1.0f, A[0], N, B[0], K,
        0.0f, C[0], K);
    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t_end - t_start;
    // printf("Dense×Dense (sgemm): %.6f s\n",
    //        diff.count());
    // print_matrix(A, M, N);
    // print_matrix(B, N, K);
    // print_matrix(C, M, K);
    free2float(A); free2float(B); free2float(C);
    return diff.count();

}