#include "../../include/MKL_Sparse_Methods.h"
#include "../../include/csr_builder.h"
#include <cstring>
#include <thread>
#include <algorithm>
#include <mutex>
#include <future>
#include <memory>
#include <immintrin.h>
#include <cstdint>
#include <vector>

// ------------------------ Helpers ------------------------
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

#if defined(_MSC_VER)
static inline void prefetch_read(const void* p) { _mm_prefetch((const char*)p, _MM_HINT_T0); }
#else
static inline void prefetch_read(const void* p) { __builtin_prefetch(p, 0, 1); }
#endif

// ------------------------ Micro-kernels (AVX2) ------------------------
static inline void microkernel_j32_accumulate_row(
    float* __restrict acc_row,
    const float* __restrict Bp_row0,   // 注意：可设置为 (Bslab + j - kb*nb_eff)
    int nb_eff,
    const MKL_INT* __restrict col_idx,
    const float*   __restrict val,
    MKL_INT it0, MKL_INT it1,
    int j0
) {
    __m256 acc0 = _mm256_load_ps(acc_row + j0 +  0);
    __m256 acc1 = _mm256_load_ps(acc_row + j0 +  8);
    __m256 acc2 = _mm256_load_ps(acc_row + j0 + 16);
    __m256 acc3 = _mm256_load_ps(acc_row + j0 + 24);

    for (MKL_INT p = it0; p < it1; ++p) {
        const MKL_INT k = col_idx[p];
        const float   v = val[p];
        const float* __restrict Brow = Bp_row0 + (size_t)k * (size_t)nb_eff;
        if (p + 8 < it1) {
            prefetch_read(&col_idx[p + 8]);
            prefetch_read(&val[p + 8]);
        }
        __m256 vv = _mm256_set1_ps(v);
        __m256 b0 = _mm256_loadu_ps(Brow +  0);
        __m256 b1 = _mm256_loadu_ps(Brow +  8);
        __m256 b2 = _mm256_loadu_ps(Brow + 16);
        __m256 b3 = _mm256_loadu_ps(Brow + 24);
        acc0 = _mm256_fmadd_ps(vv, b0, acc0);
        acc1 = _mm256_fmadd_ps(vv, b1, acc1);
        acc2 = _mm256_fmadd_ps(vv, b2, acc2);
        acc3 = _mm256_fmadd_ps(vv, b3, acc3);
    }
    _mm256_store_ps(acc_row + j0 +  0, acc0);
    _mm256_store_ps(acc_row + j0 +  8, acc1);
    _mm256_store_ps(acc_row + j0 + 16, acc2);
    _mm256_store_ps(acc_row + j0 + 24, acc3);
}

static inline void microkernel_j8_accumulate_row(
    float* __restrict acc_row,
    const float* __restrict Bp_row0,
    int nb_eff,
    const MKL_INT* __restrict col_idx,
    const float*   __restrict val,
    MKL_INT it0, MKL_INT it1,
    int j0
) {
    __m256 acc = _mm256_load_ps(acc_row + j0);
    for (MKL_INT p = it0; p < it1; ++p) {
        const MKL_INT k = col_idx[p];
        const float   v = val[p];
        const float* __restrict Brow = Bp_row0 + (size_t)k * (size_t)nb_eff + j0;
        __m256 vv = _mm256_set1_ps(v);
        __m256 b  = _mm256_loadu_ps(Brow);
        acc = _mm256_fmadd_ps(vv, b, acc);
    }
    _mm256_store_ps(acc_row + j0, acc);
}

// ------------------------ 主函数 ------------------------
bool MKL_Sparse_CooXDense_Fast_Gustavson_optimized(
    float* denseA,
    float* denseB,
    float* denseC,
    int rowsA,
    int colsA,
    int colsC,
    int /*flag*/)
{
    // 1) 构建 CSR
    std::vector<MKL_INT> row_ptr;
    std::vector<MKL_INT> col_idx;
    std::vector<float>   val;
    const int builder_threads = 10;
    if (!build_csr_from_denseA(denseA, rowsA, colsA, row_ptr, col_idx, val, builder_threads)) {
        fprintf(stderr, "build_csr_from_denseA failed\n");
        return false;
    }

    // 2) 清零 C（保持原有行为）
    std::memset(denseC, 0, sizeof(float) * (size_t)rowsA * (size_t)colsC);

    // 3) tile 参数
    const double util_ratio   = 0.5;
    const int    target_bytes = static_cast<int>(L2_BYTES * util_ratio);
    int Kc = 384;                               // K slab
    int Nb = std::min(colsC, 64);               // N tile
    int Rb = std::max(1, target_bytes / (4 * std::max(1, Nb)) - Kc);
    if (Rb > rowsA) Rb = rowsA;
    if (Rb < 1) {
        Kc = std::max<int>(8, target_bytes / (4 * std::max(1, Nb)) - 1);
        Rb = std::max<int>(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    }
    Kc = std::max(8, std::min(Kc, colsA));
    Nb = std::max(16, std::min(Nb, colsC));
    Rb = std::max(1, std::min(Rb, rowsA));

    // 4) 线程池
    int num_threads = builder_threads;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (num_threads <= 0) num_threads = (int)hw;
    if (num_threads > rowsA) num_threads = rowsA;

    static std::unique_ptr<ThreadPool> pool;
    static std::mutex pool_init_mtx;
    {
        std::lock_guard<std::mutex> lk(pool_init_mtx);
        if (!pool) pool.reset(new ThreadPool((size_t)num_threads));
    }

    // 5) 以 cb 为任务单元：对每个 cb，只为“活跃”的 K slab 打包并复用到所有 rb
    std::vector<std::future<void>> futures;
    futures.reserve((colsC + Nb - 1) / Nb);

    for (int cb = 0; cb < colsC; cb += Nb) {
        const int nb_eff = std::min(Nb, colsC - cb);
        futures.push_back(pool->enqueue([=, &row_ptr, &col_idx, &val, &denseB, &denseC]() {
            const int VEC_WIDTH = 8;  // AVX2
            const int vec_end   = (nb_eff / VEC_WIDTH) * VEC_WIDTH;
            const int acc_elems_per_row = ((nb_eff + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;
            const int JT = 4 * VEC_WIDTH; // 32 floats

            // ---- 线程本地 acc_tile（最大 Rb×acc_elems_per_row）----
            const size_t acc_tile_max_elems = (size_t)std::min(Rb, rowsA) * (size_t)acc_elems_per_row;
            void* acc_tile_tmp = portable_aligned_alloc(ACC_ALIGN, acc_tile_max_elems * sizeof(float));
            bool acc_tile_portable = true;
            if (!acc_tile_tmp) { acc_tile_tmp = malloc(acc_tile_max_elems * sizeof(float)); acc_tile_portable = false; if (!acc_tile_tmp) return; }
            float* __restrict acc_tile_buf = (float*)acc_tile_tmp;

            // ---- B slab 缓冲：仅 Kc×nb_eff，按需打包 ----
            const size_t bslab_elems = (size_t)Kc * (size_t)nb_eff;
            void* bslab_tmp = portable_aligned_alloc(ACC_ALIGN, bslab_elems * sizeof(float));
            bool bslab_portable = true;
            if (!bslab_tmp) { bslab_tmp = malloc(bslab_elems * sizeof(float)); bslab_portable = false; if (!bslab_tmp) { if (acc_tile_tmp) { if (acc_tile_portable) portable_aligned_free(acc_tile_tmp); else free(acc_tile_tmp);} return; } }
            float* __restrict Bslab = (float*)bslab_tmp;

            // ---- per-row 游标（跨 kb 递进）和行末 ----
            std::vector<MKL_INT> pos((size_t)rowsA), rend((size_t)rowsA);
            for (int i = 0; i < rowsA; ++i) { pos[i] = row_ptr[i]; rend[i] = row_ptr[i + 1]; }

            // ---- 预留 it0/it1（每个 kb 计算一次，rb 重用）----
            std::vector<MKL_INT> it0((size_t)rowsA), it1((size_t)rowsA);

            // ---- 外层: K slab（只对活跃 slab 打包）----
            for (int kb = 0; kb < colsA; kb += Kc) {
                const int kc_eff = std::min(Kc, colsA - kb);
                const int kblock_end = kb + kc_eff;

                // 先扫描所有行，求出 it0/it1，并统计是否“活跃”
                bool active = false;
                for (int i = 0; i < rowsA; ++i) {
                    MKL_INT p = pos[i];
                    const MKL_INT end = rend[i];
                    while (p < end && col_idx[p] < kb) ++p;          // 跳到 >= kb
                    MKL_INT q = p;
                    while (q < end && col_idx[q] < kblock_end) ++q;  // 到 < kb+kc
                    it0[i] = p; it1[i] = q;
                    if (p < q) active = true;
                }
                if (!active) {
                    // 本 slab 完全没有非零，推进 pos 后跳过
                    for (int i = 0; i < rowsA; ++i) pos[i] = it1[i];
                    continue;
                }

                // 打包本 slab：kc_eff×nb_eff（只打一次）
                for (int kk = 0; kk < kc_eff; ++kk) {
                    const float* __restrict Brow = denseB + (size_t)(kb + kk) * (size_t)colsC + (size_t)cb;
                    float*       __restrict Bdst = Bslab  + (size_t)kk * (size_t)nb_eff;
                    if (nb_eff > 0) std::memcpy(Bdst, Brow, sizeof(float) * (size_t)nb_eff);
                }

                // 计算：对所有 rb，使用同一个 Bslab
                for (int rb = 0; rb < rowsA; rb += Rb) {
                    const int rb_eff = std::min(Rb, rowsA - rb);
                    if (rb_eff <= 0) continue;

                    float* __restrict acc_tile = acc_tile_buf;

                    // acc_tile 清零（我们已在函数开头把 C 清零，这里不再从 C memcpy）
                    // 置零 padded 整行，利于对齐向量化
                    for (int local_i = 0; local_i < rb_eff; ++local_i) {
                        float* __restrict dest_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;
                        std::memset(dest_row, 0, sizeof(float) * (size_t)acc_elems_per_row);
                    }

                    // 以行为单位做累加（寄存器微核），范围为 it0/it1
                    const float* __restrict Bp_offset_base = Bslab - (size_t)kb * (size_t)nb_eff; // 让 Brow = (Bp_offset_base + k*nb + j)
                    for (int local_i = 0; local_i < rb_eff; ++local_i) {
                        const int i = rb + local_i;
                        const MKL_INT p0 = it0[i], p1 = it1[i];
                        if (p0 >= p1) continue;

                        float* __restrict acc_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;

                        int j = 0;
                        const int j32_end = (vec_end / JT) * JT;
                        for (; j < j32_end; j += JT) {
                            const float* __restrict Bp_row0 = Bp_offset_base + (size_t)j;
                            microkernel_j32_accumulate_row(acc_row, Bp_row0, nb_eff, col_idx.data(), val.data(), p0, p1, j);
                        }
                        for (; j + VEC_WIDTH - 1 < vec_end; j += VEC_WIDTH) {
                            const float* __restrict Bp_row0 = Bp_offset_base + (size_t)j;
                            microkernel_j8_accumulate_row(acc_row, Bp_row0, nb_eff, col_idx.data(), val.data(), p0, p1, j);
                        }
                        for (; j < nb_eff; ++j) {
                            float acc = acc_row[j];
                            for (MKL_INT q = p0; q < p1; ++q) {
                                const MKL_INT k = col_idx[q];
                                const float   v = val[q];
                                acc += v * Bslab[(size_t)(k - kb) * (size_t)nb_eff + (size_t)j];
                            }
                            acc_row[j] = acc;
                        }
                    } // rows in rb

                    // 写回：把本 slab 贡献加到 C（C 初始为 0，等价于最终累加）
                    for (int local_i = 0; local_i < rb_eff; ++local_i) {
                        const int i = rb + local_i;
                        float* __restrict Crow   = denseC + (size_t)i * (size_t)colsC + (size_t)cb;
                        float* __restrict srcrow = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;

                        int j = 0;
                        for (; j + 7 < nb_eff; j += 8) {
                            __m256 c  = _mm256_loadu_ps(Crow + j);
                            __m256 v  = _mm256_load_ps(srcrow + j);
                            c = _mm256_add_ps(c, v);
                            _mm256_storeu_ps(Crow + j, c);
                        }
                        for (; j < nb_eff; ++j) Crow[j] += srcrow[j];
                    }
                } // rb

                // 推进 per-row 游标
                for (int i = 0; i < rowsA; ++i) pos[i] = it1[i];
            } // kb

            if (bslab_tmp)   { if (bslab_portable)   portable_aligned_free(bslab_tmp);   else free(bslab_tmp); }
            if (acc_tile_tmp){ if (acc_tile_portable)portable_aligned_free(acc_tile_tmp); else free(acc_tile_tmp); }
        }));
    }

    for (auto& f : futures) f.get();
    return true;
}
