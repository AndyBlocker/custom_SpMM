#include "../include/MKL_Sparse_Methods.h"
#include <ctime>
#include <thread>
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include "../include/csr_builder.h"

// Portable aligned alloc/free helpers
static inline void* portable_aligned_alloc(size_t alignment, size_t size) {
    if (size == 0) return nullptr;
    size_t extra = alignment - 1 + sizeof(void*);
    void* raw = malloc(size + extra);
    if (!raw) return nullptr;
    uintptr_t raw_addr = (uintptr_t)raw + sizeof(void*);
    uintptr_t aligned = (raw_addr + (alignment - 1)) & ~(alignment - 1);
    void** store = (void**)(aligned - sizeof(void*));
    *store = raw;
    return (void*)aligned;
}

static inline void portable_aligned_free(void* p) {
    if (!p) return;
    void** store = (void**)((uintptr_t)p - sizeof(void*));
    void* raw = *store;
    free(raw);
}

// Prefetch helper
#if defined(_MSC_VER)
static inline void prefetch_read(const void* p) { _mm_prefetch((const char*)p, _MM_HINT_T0); }
#else
static inline void prefetch_read(const void* p) { __builtin_prefetch(p, 0, 1); }
#endif
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

    // Debug print removed for performance
    // if (tid == 17) {
    //     printf("tid=%d\n", tid);
    // }

    // 1) 拆分成 X×Y 网格
    int tx = nt / 16;
    // Ensure tx is at least 1 to avoid division by zero
    if (tx < 1) tx = 1;
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

// Optimized implementation with AVX2 and cache tiling
bool MKL_Sparse_CooXDense_Fast_Gustavson_new_yk(
    float* denseA,
    float* denseB,
    float* denseC,
    int rowsA,
    int colsA,
    int colsC,
    int /*flag*/)
{
    // 1) build CSR
    std::vector<MKL_INT> row_ptr;
    std::vector<MKL_INT> col_idx;
    std::vector<float> val;

    bool ok = build_csr_from_denseA(denseA, rowsA, colsA, row_ptr, col_idx, val, /*num_threads=*/10);
    if (!ok) {
        fprintf(stderr, "build_csr_from_denseA failed\n");
        return false;
    }

    int nnz = (int)row_ptr[rowsA];

    // 2) zero C (keeps original behavior)
    std::memset(denseC, 0, sizeof(float) * (size_t)rowsA * (size_t)colsC);

    // 3) tiling params
    const double util_ratio = 0.5;
    const int target_bytes = static_cast<int>(L2_BYTES * util_ratio);
    int Kc = 384, Nb = std::min(colsC, 64);
    int Rb = std::max(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    if (Rb > rowsA) Rb = rowsA;
    if (Rb < 1) {
        Kc = std::max<int>(8, target_bytes / (4 * std::max(1, Nb)) - 1);
        Rb = std::max<int>(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    }
    Kc = std::max(8, std::min(Kc, colsA));
    Nb = std::max(16, std::min(Nb, colsC));
    Rb = std::max(1, std::min(Rb, rowsA));

    // multi-thread: choose thread count
    int num_threads = 10;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (num_threads <= 0) num_threads = (int)hw;
    if (num_threads > rowsA) num_threads = rowsA;

    // persistent thread pool (static so it's reused across calls)
    static std::unique_ptr<ThreadPool> pool;
    static std::mutex pool_init_mtx;
    {
        std::lock_guard<std::mutex> lk(pool_init_mtx);
        if (!pool) pool.reset(new ThreadPool((size_t)num_threads));
    }

    // Phase 1: parallel count nonzeros per row -> rowCounts
    std::vector<int> rowCounts((size_t)rowsA, 0);

    int chunk = (rowsA + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int cb = 0; cb < colsC; cb += Nb) {
        int nb_eff = std::min(Nb, colsC - cb);
        futures.push_back(pool->enqueue([=, &row_ptr, &col_idx, &val, &denseC]() {
            int vec_end = (nb_eff / VEC_WIDTH) * VEC_WIDTH;
            int acc_elems_per_row = ((nb_eff + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;

            // allocate a reusable acc_tile buffer sized for maximum rb (Rb) for this cb
            size_t max_rb_eff = (size_t)std::min(Rb, rowsA);
            size_t acc_tile_max_elems = max_rb_eff * (size_t)acc_elems_per_row;
            void* acc_tile_buf_tmp = portable_aligned_alloc(ACC_ALIGN, acc_tile_max_elems * sizeof(float));
            bool acc_tile_buf_portable = true;
            if (!acc_tile_buf_tmp) {
                acc_tile_buf_tmp = malloc(acc_tile_max_elems * sizeof(float));
                acc_tile_buf_portable = false;
            }
            float* acc_tile_buf = (float*)acc_tile_buf_tmp;

            // iterate over rb blocks
            for (int rb = 0; rb < rowsA; rb += Rb) {
                int rb_eff = std::min(Rb, rowsA - rb);
                if (rb_eff <= 0) continue;

                // acc_tile for this rb is the prefix of acc_tile_buf
                float* acc_tile = acc_tile_buf; // size = rb_eff * acc_elems_per_row floats

                // initialization: COPY denseC small block into acc_tile (only once per (cb,rb))
                for (int local_i = 0; local_i < rb_eff; ++local_i) {
                    int i = rb + local_i;
                    float* Crow = denseC + (size_t)i * (size_t)colsC + (size_t)cb;
                    float* dest_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;
                    // copy nb_eff floats
                    if (nb_eff > 0) memcpy(dest_row, Crow, sizeof(float) * (size_t)nb_eff);
                    // zero padding up to acc_elems_per_row (small <= VEC_WIDTH)
                    for (int t = nb_eff; t < acc_elems_per_row; ++t) dest_row[t] = 0.0f;
                }

                // now for this (cb,rb) iterate over kb and accumulate directly into acc_tile
                for (int kb = 0; kb < colsA; kb += Kc) {
                    int kc_eff = std::min(Kc, colsA - kb);

                    // pack denseB for this (kb,cb)
                    size_t pack_elems = (size_t)kc_eff * (size_t)nb_eff;
                    void* packed_tmp = portable_aligned_alloc(ACC_ALIGN, pack_elems * sizeof(float));
                    bool packed_portable = true;
                    if (!packed_tmp) {
                        packed_tmp = malloc(pack_elems * sizeof(float));
                        packed_portable = false;
                    }
                    float* packedB = (float*)packed_tmp;
                    for (int kk = 0; kk < kc_eff; ++kk) {
                        const float* Brow = denseB + (size_t)(kb + kk) * (size_t)colsC + (size_t)cb;
                        float* dest = packedB + (size_t)kk * (size_t)nb_eff;
                        if (nb_eff > 0) memcpy(dest, Brow, sizeof(float) * (size_t)nb_eff);
                    }

                    // accumulate: for each row in rb, update its acc_row using packedB
                    for (int local_i = 0; local_i < rb_eff; ++local_i) {
                        int i = rb + local_i;
                        float* acc_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;

                        MKL_INT start = row_ptr[i], end = row_ptr[i + 1];
                        if (start >= end) continue;

                        // it0/it1 within current kb block
                        MKL_INT it0 = start;
                        while (it0 < end && col_idx[it0] < kb) ++it0;
                        MKL_INT it1 = it0;
                        MKL_INT kblock_end = kb + kc_eff;
                        while (it1 < end && col_idx[it1] < kblock_end) ++it1;
                        if (it0 >= it1) continue;

                        MKL_INT p = it0;
                        // 4-wide unroll
                        for (; p + 3 < it1; p += 4) {
                            MKL_INT k0 = col_idx[p + 0]; float v0 = val[p + 0];
                            MKL_INT k1 = col_idx[p + 1]; float v1 = val[p + 1];
                            MKL_INT k2 = col_idx[p + 2]; float v2 = val[p + 2];
                            MKL_INT k3 = col_idx[p + 3]; float v3 = val[p + 3];

                            const float* B0 = packedB + (size_t)(k0 - kb) * (size_t)nb_eff;
                            const float* B1 = packedB + (size_t)(k1 - kb) * (size_t)nb_eff;
                            const float* B2 = packedB + (size_t)(k2 - kb) * (size_t)nb_eff;
                            const float* B3 = packedB + (size_t)(k3 - kb) * (size_t)nb_eff;

                            __m256 v0v = _mm256_set1_ps(v0);
                            __m256 v1v = _mm256_set1_ps(v1);
                            __m256 v2v = _mm256_set1_ps(v2);
                            __m256 v3v = _mm256_set1_ps(v3);

                            int j = 0;
                            for (; j + VEC_WIDTH3 < vec_end; j += VEC_WIDTH4) {
                                __m256 acc0 = _mm256_load_ps(acc_row + j);
                                __m256 acc1 = _mm256_load_ps(acc_row + j + VEC_WIDTH);
                                __m256 acc2 = _mm256_load_ps(acc_row + j + VEC_WIDTH * 2);
                                __m256 acc3 = _mm256_load_ps(acc_row + j + VEC_WIDTH * 3);

                                __m256 b0_0 = _mm256_load_ps(B0 + j);
                                __m256 b1_0 = _mm256_load_ps(B1 + j);
                                __m256 b2_0 = _mm256_load_ps(B2 + j);
                                __m256 b3_0 = _mm256_load_ps(B3 + j);

                                __m256 b0_1 = _mm256_load_ps(B0 + j + VEC_WIDTH);
                                __m256 b1_1 = _mm256_load_ps(B1 + j + VEC_WIDTH);
                                __m256 b2_1 = _mm256_load_ps(B2 + j + VEC_WIDTH);
                                __m256 b3_1 = _mm256_load_ps(B3 + j + VEC_WIDTH);

                                __m256 b0_2 = _mm256_load_ps(B0 + j + VEC_WIDTH * 2);
                                __m256 b1_2 = _mm256_load_ps(B1 + j + VEC_WIDTH * 2);
                                __m256 b2_2 = _mm256_load_ps(B2 + j + VEC_WIDTH * 2);
                                __m256 b3_2 = _mm256_load_ps(B3 + j + VEC_WIDTH * 2);

                                __m256 b0_3 = _mm256_load_ps(B0 + j + VEC_WIDTH * 3);
                                __m256 b1_3 = _mm256_load_ps(B1 + j + VEC_WIDTH * 3);
                                __m256 b2_3 = _mm256_load_ps(B2 + j + VEC_WIDTH * 3);
                                __m256 b3_3 = _mm256_load_ps(B3 + j + VEC_WIDTH * 3);

                                acc0 = _mm256_fmadd_ps(v0v, b0_0, acc0);
                                acc1 = _mm256_fmadd_ps(v0v, b0_1, acc1);
                                acc2 = _mm256_fmadd_ps(v0v, b0_2, acc2);
                                acc3 = _mm256_fmadd_ps(v0v, b0_3, acc3);

                                acc0 = _mm256_fmadd_ps(v1v, b1_0, acc0);
                                acc1 = _mm256_fmadd_ps(v1v, b1_1, acc1);
                                acc2 = _mm256_fmadd_ps(v1v, b1_2, acc2);
                                acc3 = _mm256_fmadd_ps(v1v, b1_3, acc3);

                                acc0 = _mm256_fmadd_ps(v2v, b2_0, acc0);
                                acc1 = _mm256_fmadd_ps(v2v, b2_1, acc1);
                                acc2 = _mm256_fmadd_ps(v2v, b2_2, acc2);
                                acc3 = _mm256_fmadd_ps(v2v, b2_3, acc3);

                                acc0 = _mm256_fmadd_ps(v3v, b3_0, acc0);
                                acc1 = _mm256_fmadd_ps(v3v, b3_1, acc1);
                                acc2 = _mm256_fmadd_ps(v3v, b3_2, acc2);
                                acc3 = _mm256_fmadd_ps(v3v, b3_3, acc3);

                                _mm256_store_ps(acc_row + j, acc0);
                                _mm256_store_ps(acc_row + j + VEC_WIDTH, acc1);
                                _mm256_store_ps(acc_row + j + VEC_WIDTH * 2, acc2);
                                _mm256_store_ps(acc_row + j + VEC_WIDTH * 3, acc3);

                                prefetch_read(B0 + j + 16);
                                prefetch_read(B1 + j + 16);
                                prefetch_read(B2 + j + 16);
                                prefetch_read(B3 + j + 16);
                            }
                            for (; j < vec_end; j += VEC_WIDTH) {
                                __m256 accv = _mm256_load_ps(acc_row + j);
                                __m256 b0 = _mm256_load_ps(B0 + j);
                                __m256 b1 = _mm256_load_ps(B1 + j);
                                __m256 b2 = _mm256_load_ps(B2 + j);
                                __m256 b3 = _mm256_load_ps(B3 + j);
                                accv = _mm256_fmadd_ps(v0v, b0, accv);
                                accv = _mm256_fmadd_ps(v1v, b1, accv);
                                accv = _mm256_fmadd_ps(v2v, b2, accv);
                                accv = _mm256_fmadd_ps(v3v, b3, accv);
                                _mm256_store_ps(acc_row + j, accv);
                            }
                        } // end p by 4

                        // singletons
                        for (; p < it1; ++p) {
                            if (p + PREFETCH_P < it1) {
                                prefetch_read(&col_idx[p + PREFETCH_P]);
                                prefetch_read(&val[p + PREFETCH_P]);
                            }
                            MKL_INT k_col = col_idx[p];
                            float v = val[p];
                            const float* Brow = packedB + (size_t)(k_col - kb) * (size_t)nb_eff;
                            __m256 vv = _mm256_set1_ps(v);
                            int j = 0;
                            for (; j < vec_end; j += VEC_WIDTH) {
                                __m256 accv = _mm256_load_ps(acc_row + j);
                                __m256 bvec = _mm256_load_ps(Brow + j);
                                accv = _mm256_fmadd_ps(vv, bvec, accv);
                                _mm256_store_ps(acc_row + j, accv);
                            }
                            if (vec_end < nb_eff) {
                                for (int jj = vec_end; jj < nb_eff; ++jj) acc_row[jj] += v * Brow[jj];
                            }
                        } // end singletons
                    } // end rows in rb

                    // free packedB for this kb
                    if (packed_tmp) { if (packed_portable) portable_aligned_free(packed_tmp); else free(packed_tmp); }
                } // end kb loop

                // write back acc_tile once for (cb,rb)
                for (int local_i = 0; local_i < rb_eff; ++local_i) {
                    int i = rb + local_i;
                    float* Crow = denseC + (size_t)i * (size_t)colsC + (size_t)cb;
                    float* src_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;
                    int j = 0;
                    for (; j + VEC_WIDTH - 1 < nb_eff; j += VEC_WIDTH) {
                        __m256 tmpv = _mm256_load_ps(src_row + j);    // aligned load
                        _mm256_storeu_ps(Crow + j, tmpv);            // store to possibly unaligned C
                    }
                    for (; j < nb_eff; ++j) Crow[j] = src_row[j];
                }
            } // rb loop

            // free acc_tile_buf for this cb
            if (acc_tile_buf_tmp) { if (acc_tile_buf_portable) portable_aligned_free(acc_tile_buf_tmp); else free(acc_tile_buf_tmp); }
        }));
    } // cb

    for (auto& f : futures) {
        f.get();
    }

    return true;
}