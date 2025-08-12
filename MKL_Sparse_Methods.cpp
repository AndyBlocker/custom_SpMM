#include "MKL_Sparse_Methods.h"
#include <ctime>
#include <thread>
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
/*------------------------------------------------------------
 *  内存工具：连续块 + 行指针
 *-----------------------------------------------------------*/
static void** alloc2(std::size_t n1, std::size_t n2, std::size_t sz)
{
    void** ptr = static_cast<void**>(std::malloc(n2 * sizeof(void*)));
    if (!ptr) return nullptr;
    ptr[0] = std::malloc(n1 * n2 * sz);
    if (!ptr[0]) { std::free(ptr); return nullptr; }
    for (std::size_t i = 1; i < n2; ++i)
        ptr[i] = static_cast<char*>(ptr[0]) + i * n1 * sz;
    return ptr;
}

static void free2(void** p)
{
    if (p) { std::free(p[0]); std::free(p); }
}

float** alloc2float(std::size_t cols, std::size_t rows)
{
    return reinterpret_cast<float**>(alloc2(cols, rows, sizeof(float)));
}

void free2float(float** p) { free2(reinterpret_cast<void**>(p)); }


// -------------------- CSR 构造阶段一：并行计数 --------------------
struct CSRCountArgs {
    int          tid, nt;
    int          rowsA, colsA;
    float** denseA;
    std::vector<int>* rowCounts;
};

void* csr_count_thread(void* arg) {
    auto* a = static_cast<CSRCountArgs*>(arg);
    int tid = a->tid, nt = a->nt;
    int rowsA = a->rowsA, colsA = a->colsA;
    float** A = a->denseA;
    auto& counts = *a->rowCounts;

    int chunk = (rowsA + nt - 1) / nt;
    int r0 = tid * chunk;
    int r1 = std::min(r0 + chunk, rowsA);

    for (int i = r0; i < r1; ++i) {
        int c = 0;
        float* Ai = A[i];
        for (int j = 0; j < colsA; ++j)
            if (Ai[j] != 0.0f) ++c;
        counts[i] = c;
    }
    return nullptr;
}

// -------------------- CSR 构造阶段二：并行填充 --------------------
struct CSRFillArgs {
    int                  tid, nt;
    int                  rowsA, colsA;
    float** denseA;
    const std::vector<MKL_INT>* row_ptr;
    std::vector<int>* localOffsets;
    std::vector<MKL_INT>* col_idx;
    std::vector<float>* val;
};

void* csr_fill_thread(void* arg) {
    auto* a = static_cast<CSRFillArgs*>(arg);
    int tid = a->tid, nt = a->nt;
    int rowsA = a->rowsA, colsA = a->colsA;
    float** A = a->denseA;
    auto& rp = *a->row_ptr;
    auto& loffs = *a->localOffsets;
    auto& col = *a->col_idx;
    auto& v = *a->val;

    int chunk = (rowsA + nt - 1) / nt;
    int r0 = tid * chunk;
    int r1 = std::min(r0 + chunk, rowsA);

    for (int i = r0; i < r1; ++i) {
        int base = rp[i];
        int off = 0;
        float* Ai = A[i];
        for (int j = 0; j < colsA; ++j) {
            float x = Ai[j];
            if (x != 0.0f) {
                int pos = base + off;
                col[pos] = j;
                v[pos] = x;
                ++off;
            }
        }
        loffs[i] = off;
    }
    return nullptr;
}

struct CountArgs {
    int        tid, nt;
    int        rowsA, colsA;
    float** denseA;
    std::vector<int>* rowCounts;
};

void* count_thread(void* arg) {
    auto* a = static_cast<CountArgs*>(arg);
    int tid = a->tid, nt = a->nt;
    int rowsA = a->rowsA, colsA = a->colsA;
    float** A = a->denseA;
    auto& counts = *a->rowCounts;

    int chunk = (rowsA + nt - 1) / nt;
    int r0 = tid * chunk;
    int r1 = std::min(r0 + chunk, rowsA);

    for (int i = r0; i < r1; ++i) {
        int c = 0;
        float* Ai = A[i];
        for (int j = 0; j < colsA; ++j)
            if (Ai[j] != 0.0f) ++c;
        counts[i] = c;
    }
    return nullptr;
}

// ---------- 线程参数：阶段 2（填充 CSR） ----------
struct FillArgs {
    int                     tid, nt;
    int                     rowsA, colsA;
    float** denseA;
    const std::vector<MKL_INT>* row_ptr;
    std::vector<int>* localOffsets;
    MKL_INT* col_idx;
    float* val;
};

void* fill_thread(void* arg) {
    auto* a = static_cast<FillArgs*>(arg);
    int tid = a->tid, nt = a->nt;
    int rowsA = a->rowsA, colsA = a->colsA;
    float** A = a->denseA;
    auto& rp = *a->row_ptr;
    auto& loffs = *a->localOffsets;
    MKL_INT* col = a->col_idx;
    float* v = a->val;

    int chunk = (rowsA + nt - 1) / nt;
    int r0 = tid * chunk;
    int r1 = std::min(r0 + chunk, rowsA);

    for (int i = r0; i < r1; ++i) {
        int base = rp[i];
        int off = 0;
        float* Ai = A[i];
        for (int j = 0; j < colsA; ++j) {
            float x = Ai[j];
            if (x != 0.0f) {
                int pos = base + off;
                col[pos] = j;
                v[pos] = x;
                ++off;
            }
        }
        loffs[i] = off;
    }
    return nullptr;
}



/*------------------------------------------------------------
 *  稀疏 × 稠密 乘法（核心优化实现） - 使用 std::thread
 *-----------------------------------------------------------*/
bool MKL_Sparse_CooXDense_Fast(float** denseA,
    float** denseB,
    float** denseC,
    int   rowsA,
    int   colsA,
    int   colsC,
    int   flag)
{
    //auto t_start = std::chrono::steady_clock::now();
    // 1) 阶段 1：并行统计每行 nnz
    int nt = 16;
    if (rowsA <= 1024) {
        nt = 4;
    }
    if (nt < 1) nt = 4;
    std::vector<int> rowCounts(rowsA, 0);
    // 使用 std::thread 替代 pthread_t
    std::vector<std::thread> ct(nt);
    std::vector<CountArgs> cargs(nt);
    for (int t = 0; t < nt; ++t) {
        cargs[t] = { t, nt, rowsA, colsA, denseA, &rowCounts };
        // 使用 std::thread 构造函数替代 pthread_create
        ct[t] = std::thread(count_thread, &cargs[t]);
    }
    // 使用 join 替代 pthread_join
    for (int t = 0; t < nt; ++t) {
        if (ct[t].joinable()) { // 检查线程是否可 join
            ct[t].join();
        }
    }

    // 2) 串行前缀和，构造 row_ptr
    std::vector<MKL_INT> row_ptr(rowsA + 1, 0);
    for (int i = 0; i < rowsA; ++i) {
        row_ptr[i + 1] = row_ptr[i] + rowCounts[i];
    }
    int nnz = row_ptr[rowsA];

    // 3) 分配 MKL CSR 数组
    MKL_INT* row_start = (MKL_INT*)mkl_malloc(rowsA * sizeof(MKL_INT), 64);
    MKL_INT* row_end = (MKL_INT*)mkl_malloc(rowsA * sizeof(MKL_INT), 64);
    MKL_INT* col_idx = (MKL_INT*)mkl_malloc(nnz * sizeof(MKL_INT), 64);
    float* val = (float*)mkl_malloc(nnz * sizeof(float), 64);

    if (!row_start || !row_end || !col_idx || !val) {
        fprintf(stderr, "MKL memory allocation failed\n");
        // 释放已分配的内存
        mkl_free(row_start); mkl_free(row_end);
        mkl_free(col_idx);   mkl_free(val);
        return false;
    }

    // 填 row_start / row_end
    for (int i = 0; i < rowsA; ++i) {
        row_start[i] = row_ptr[i];
        row_end[i] = row_ptr[i + 1];
    }

    // 4) 阶段 2：并行填充 col_idx & val
    std::vector<int> localOffsets(rowsA, 0);
    // 使用 std::thread 替代 pthread_t
    std::vector<std::thread> ft(nt);
    std::vector<FillArgs> fargs(nt);
    for (int t = 0; t < nt; ++t) {
        fargs[t] = { t, nt, rowsA, colsA, denseA,
                    &row_ptr, &localOffsets,
                    col_idx, val };
        // 使用 std::thread 构造函数替代 pthread_create
        ft[t] = std::thread(fill_thread, &fargs[t]);
    }
    // 使用 join 替代 pthread_join
    for (int t = 0; t < nt; ++t) {
        if (ft[t].joinable()) {
            ft[t].join();
        }
    }
    //auto t_end = std::chrono::steady_clock::now();
    //std::chrono::duration<double> diff = t_end - t_start;
    //printf("Sparse×Dense (mkl): %.6f s\n",
    //    diff.count());

    /* ---------- 2. 生成并优化 CSR 句柄 ---------- */
    sparse_matrix_t csrA;
    sparse_status_t status = mkl_sparse_s_create_csr(&csrA,
        SPARSE_INDEX_BASE_ZERO,
        rowsA,
        colsA,
        row_start,
        row_end,
        col_idx,
        val);

    if (status != SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "create_csr failed, status: %d\n", status);
        // 释放 MKL 内存
        mkl_free(row_start); mkl_free(row_end);
        mkl_free(col_idx);   mkl_free(val);
        return false;
    }

    mkl_sparse_optimize(csrA);     /* Inspector‑Executor */

    /* ---------- 3. 稠密矩阵直接使用 Row‑Major 内存 ---------- */
    const float alpha = 1.0f, beta = 0.0f;
    sparse_operation_t op = SPARSE_OPERATION_NON_TRANSPOSE;
    if (flag == 1) op = SPARSE_OPERATION_TRANSPOSE;
    else if (flag == 2) op = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;

    /* denseB[0] / denseC[0] 指向连续块；列数 = colsC */
    status = mkl_sparse_s_mm(op, alpha,
        csrA, { SPARSE_MATRIX_TYPE_GENERAL,
               SPARSE_FILL_MODE_FULL,
               SPARSE_DIAG_NON_UNIT },
        SPARSE_LAYOUT_ROW_MAJOR,
        denseB[0],          /* B */
        colsC,              /* columns */
        colsC,              /* ldx */
        beta,
        denseC[0],          /* C */
        colsC               /* ldy */
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "mkl_sparse_s_mm failed, status: %d\n", status);
        mkl_sparse_destroy(csrA); // 销毁 csrA
        mkl_free(row_start); mkl_free(row_end);
        mkl_free(col_idx);   mkl_free(val);
        return false;
    }

    /* ---------- 4. 资源释放 ---------- */
    mkl_sparse_destroy(csrA);
    mkl_free(row_start); mkl_free(row_end);
    mkl_free(col_idx);   mkl_free(val);

    return true;
}

struct GustavsonArgs {
    int    tid;
    int    num_threads;
    const std::vector<MKL_INT>* row_ptr;
    const std::vector<MKL_INT>* col_idx;
    const std::vector<float>* val;
    int    colsC;
    float** denseB;
    float** denseC;
};

void* gustavson_thread(void* arg) {
    auto* a = static_cast<GustavsonArgs*>(arg);
    int tid = a->tid;
    int nt = a->num_threads;
    auto& row_ptr = *a->row_ptr;
    auto& col_idx = *a->col_idx;
    auto& val = *a->val;
    int colsC = a->colsC;
    float** denseB = a->denseB;
    float** denseC = a->denseC;
    int rowsA = int(row_ptr.size()) - 1;

    if (tid == 17) {
        printf("tid=%d\n", tid);
    }

    // 1) 拆分成 X×Y 网格
    int tx = nt / 16;
    //if (rowsA <= 64) {
    //    tx = 1;
    //}
    while (tx > 1 && nt % tx != 0) {
        --tx;
    }
    int ty = nt / tx;

    int tid_x = tid % tx;  // 列块编号
    int tid_y = tid / tx;  // 行块编号

    // 2) 计算行范围 [r0, r1)
    int chunk_y = (rowsA + ty - 1) / ty;
    int r0 = tid_y * chunk_y;
    int r1 = std::min(r0 + chunk_y, rowsA);

    // 3) 计算列范围 [c0, c1)
    int chunk_x = (colsC + tx - 1) / tx;
    int c0 = tid_x * chunk_x;
    int c1 = std::min(c0 + chunk_x, colsC);
    int w = c1 - c0;        // 本线程负责的列宽度

    std::vector<float> Ci_buf(w);

    // 4) Gustavson on block [r0,r1)×[c0,c1)
    for (int i = r0; i < r1; ++i) {
        // 4.1 清零本列块
        std::fill(Ci_buf.begin(), Ci_buf.end(), 0.0f);

        // 4.2 遍历稀疏行，累加到本列块
        for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            int k_col = col_idx[idx];
            float v = val[idx];
            float* Bk = denseB[0] + k_col * colsC + c0;  // 列偏移到 c0

            for (int j = 0; j < w; ++j) {
                Ci_buf[j] += v * Bk[j];
            }
        }

        // 4.3 将本列块写回 C
        float* Ci = denseC[0] + i * colsC + c0;
        std::copy(Ci_buf.begin(), Ci_buf.end(), Ci);
    }

    return nullptr;
}


bool MKL_Sparse_CooXDense_Fast_Gustavson(
    float** denseA,
    float** denseB,
    float** denseC,
    int             rowsA,
    int             colsA,
    int             colsC,
    int             flag)
{
    // 1. 构造 CSR
    int nt = 16;
    if (rowsA <= 64) {
        nt = 1;
    }
    else if (rowsA <= 1024) {
        nt = 4;
    }
    if (nt < 1) nt = 4;
    std::vector<int> rowCounts(rowsA, 0);
    // 使用 std::thread 替代 pthread_t
    std::vector<std::thread> ct(nt);
    std::vector<CSRCountArgs> cargs(nt);
    for (int t = 0; t < nt; ++t) {
        cargs[t] = { t, nt, rowsA, colsA, denseA, &rowCounts };
        // 使用 std::thread 构造函数替代 pthread_create
        ct[t] = std::thread(csr_count_thread, &cargs[t]);
    }
    // 使用 join 替代 pthread_join
    for (int t = 0; t < nt; ++t) {
        if (ct[t].joinable()) {
            ct[t].join();
        }
    }

    // --- 2. 串行前缀和，生成 row_ptr ---
    std::vector<MKL_INT> row_ptr(rowsA + 1, 0);
    for (int i = 0; i < rowsA; ++i)
        row_ptr[i + 1] = row_ptr[i] + rowCounts[i];
    int nnz = row_ptr[rowsA];

    // --- 3. 并行填充 col_idx & val ---
    std::vector<MKL_INT> col_idx(nnz);
    std::vector<float>   val(nnz);
    std::vector<int>     localOffsets(rowsA, 0);
    // 使用 std::thread 替代 pthread_t
    std::vector<std::thread> ft(nt);
    std::vector<CSRFillArgs> fargs(nt);
    for (int t = 0; t < nt; ++t) {
        fargs[t] = { t, nt, rowsA, colsA, denseA,
                    &row_ptr, &localOffsets,
                    &col_idx, &val };
        // 使用 std::thread 构造函数替代 pthread_create
        ft[t] = std::thread(csr_fill_thread, &fargs[t]);
    }
    // 使用 join 替代 pthread_join
    for (int t = 0; t < nt; ++t) {
        if (ft[t].joinable()) {
            ft[t].join();
        }
    }

    // 2. 并行 Gustavson（二维线程网格）
    int num_threads = 128;
    if (rowsA <= 64) {
        num_threads = 1;
    }
    else if (rowsA <= 1024) {
        num_threads = 128;
    }

    // 使用 std::thread 替代 pthread_t
    std::vector<std::thread>     threads(num_threads);
    std::vector<GustavsonArgs> args(num_threads);

    if (num_threads <= 1)
    {
        args[0] = {
            0, num_threads,
            &row_ptr, &col_idx, &val,
            colsC, denseB, denseC
        };
        gustavson_thread(&args[0]); // 直接调用，不创建线程
    }
    else {
        for (int t = 0; t < num_threads; ++t) {
            args[t] = {
                t, num_threads,
                &row_ptr, &col_idx, &val,
                colsC, denseB, denseC
            };
            // 使用 std::thread 构造函数替代 pthread_create
            threads[t] = std::thread(gustavson_thread, &args[t]);
        }
        // 使用 join 替代 pthread_join
        for (int t = 0; t < num_threads; ++t) {
            if (threads[t].joinable()) {
                threads[t].join();
            }
        }
    }
    // row_ptr, col_idx, val, localOffsets 等 vector 会在函数结束时自动析构
    return true;
}