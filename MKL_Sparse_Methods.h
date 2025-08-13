#pragma once
/*  Intel MKL‑based sparse × dense 加速工具
 *  2025‑07‑21  重写 / 精简版
 *  关键特性：
 *    • 直接构建 CSR，避免 COO→CSR 二次转换
 *    • 只使用 ROW_MAJOR 布局，彻底去掉 B/C 的 2‑D→1‑D 冗余复制
 *    • 通过 mkl_sparse_optimize() 让 MKL 执行前期分析/调度
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mkl.h>
#include <mkl_spblas.h>

#define MAX_THREAD 16
#define LWIDTH 4
#define LWIDTH_POW2 16
constexpr int L1_BYTES = 32 * 1024;
constexpr int L2_BYTES = 1024 * 1024;
constexpr int VEC_WIDTH = 8;         // AVX2: 8 floats
constexpr int VEC_WIDTH3 = 24;         // AVX2: 8 floats
constexpr int VEC_WIDTH4 = 32;         // AVX2: 8 floats
constexpr size_t ACC_ALIGN = 64;
constexpr int PREFETCH_P = 4;

 /* ---------- 通用 2‑D Row‑Major 内存工具 ---------- */
float** alloc2float(std::size_t cols, std::size_t rows);   // cols = 列数；rows = 行数
void   free2float(float** array);

/* ---------- 稀疏 × 稠密 乘法（优化版）---------------
 * ia, ja, a 为 0‑based COO；denseB/denseC 为 Row‑Major（即 C‑style）二维指针
 * flag: 0 = op(A)=A，1 = Aᵀ，2 = Aᴴ
 * 只计算一次乘法后即释放内部资源；多次复用时请把 csrA 的创建/优化提取到外层 */
bool MKL_Sparse_CooXDense_Fast(
    float** denseA,  /* shape: rowsA × colsA */
    float** denseB,  /* shape: colsA × colsC */
    float** denseC,  /* shape: rowsA × colsC */
    int   rowsA,
    int   colsA,
    int   colsC,
    int   flag);     /* 0/1/2 同上 */

bool MKL_Sparse_CooXDense_Fast_Gustavson(
    float** denseA,  /* shape: rowsA × colsA */
    float** denseB,  /* shape: colsA × colsC */
    float** denseC,  /* shape: rowsA × colsC */
    int   rowsA,
    int   colsA,
    int   colsC,
    int   flag);     /* 0/1/2 同上 */

bool MKL_Sparse_CooXDense_Fast_Gustavson_new(
    float* denseA,
    float* denseB,
    float* denseC,
    int             rowsA,
    int             colsA,
    int             colsC,
    int             flag);


bool MKL_Sparse_CooXDense_Fast_Gustavson_new_yk(
    float* denseA,
    float* denseB,
    float* denseC,
    int             rowsA,
    int             colsA,
    int             colsC,
    int             flag);

struct Tiling {
    int Kc;
    int Nb;
    int Rb;
    int num_threads;
};

Tiling compute_tiling(int rowsA, int colsA, int colsC,
    int prefer_threads,
    double util_ratio,
    int L2_bytes,
    int vec_w);