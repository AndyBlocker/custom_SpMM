// optimized: acc_tile init moved outside kb loop, memcpy copy, reuse acc_tile per (cb,rb)
#include "MKL_Sparse_Methods.h"
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

// portable aligned alloc/free
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

bool MKL_Sparse_CooXDense_Fast_Gustavson_new_yk(
    float* denseA,
    float* denseB,
    float* denseC,
    int rowsA,
    int colsA,
    int colsC,
    int /*flag*/)
{
    // auto t_start = std::chrono::steady_clock::now();
    // 1) build CSR (same as before)
    std::vector<int> rowCounts(rowsA, 0);
#pragma omp parallel for num_threads(10) schedule(static, 2)
    for (int i = 0; i < rowsA; ++i) {
        int c = 0;
        float* Ai = denseA + (size_t)i * colsA;
        for (int j = 0; j < colsA; ++j) {
            if (Ai[j] != 0.0f) ++c;
        }
        rowCounts[i] = c;
    }
    std::vector<MKL_INT> row_ptr(rowsA + 1, 0);
    for (int i = 0; i < rowsA; ++i) row_ptr[i + 1] = row_ptr[i] + rowCounts[i];
    MKL_INT nnz = row_ptr[rowsA];
    std::vector<MKL_INT> col_idx((size_t)nnz);
    std::vector<float> val((size_t)nnz);
#pragma omp parallel for num_threads(10) schedule(static, 4)
    for (int i = 0; i < rowsA; ++i) {
        int base = row_ptr[i], off = 0;
        float* Ai = denseA + (size_t)i * colsA;
        for (int j = 0; j < colsA; ++j) {
            float x = Ai[j];
            if (x != 0.0f) { col_idx[base + off] = (MKL_INT)j; val[base + off] = x; ++off; }
        }
    }
    // auto t_end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = t_end - t_start;
    // printf("Sparse×Dense (mkl): %.6f s\n",
    //    diff.count());
    // 2) zero C
    // std::memset(denseC, 0, sizeof(float) * (size_t)rowsA * (size_t)colsC);

    // 3) tiling params
    const double util_ratio = 0.5;
    const int target_bytes = static_cast<int>(L1_BYTES * util_ratio);
    int Kc = 64, Nb = std::min(colsC, 64);
    int Rb = 64;
    if (Rb > rowsA) Rb = rowsA;
    if (Rb < 1) {
        Kc = std::max<int>(8, target_bytes / (4 * std::max(1, Nb)) - 1);
        Rb = std::max<int>(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    }
    Kc = std::max(8, std::min(Kc, colsA));
    Nb = std::max(16, std::min(Nb, colsC));
    Rb = std::max(1, std::min(Rb, rowsA));
    // printf("Kc=%d,Nb=%d,Rb=%d\n",Kc,Nb, Rb);


#pragma omp parallel for num_threads(10) schedule(dynamic) collapse(2)
    for (int rb = 0; rb < rowsA; rb += Rb) {
        for (int cb = 0; cb < colsC; cb += Nb) {
            const MKL_INT* row_ptr_p = row_ptr.data();
            const MKL_INT* col_idx_p = col_idx.data();
            const float* val_p = val.data();
            const float* denseB_p = denseB;
            float* denseC_p = denseC;

            // FIX: nb_eff must be the remaining columns in this block
            int nb_eff = std::min(Nb, colsC - cb);
            if (nb_eff <= 0) continue;

            int vec_end = (nb_eff / VEC_WIDTH) * VEC_WIDTH;
            int acc_elems_per_row = ((nb_eff + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;
            size_t max_rb_eff = (size_t)std::min(Rb, rowsA);
            size_t acc_tile_max_elems = max_rb_eff * (size_t)acc_elems_per_row;
            if (acc_tile_max_elems == 0) continue;

            // allocate acc_tile buffer for this (cb,rb) - aligned
            void* acc_tile_buf_tmp = portable_aligned_alloc(ACC_ALIGN, acc_tile_max_elems * sizeof(float));
            bool acc_tile_buf_portable = true;
            if (!acc_tile_buf_tmp) {
                acc_tile_buf_tmp = malloc(acc_tile_max_elems * sizeof(float));
                acc_tile_buf_portable = false;
                if (!acc_tile_buf_tmp) {
                    // allocation failed -> skip this tile (or handle error)
                    continue;
                }
            }
            float* acc_tile_buf = (float*)acc_tile_buf_tmp;

            // allocate packedB buffer (kc_eff * nb_eff) later per kb loop; allocate max size now to reuse
            size_t pack_elems_max = (size_t)Kc * (size_t)nb_eff;
            void* packed_tmp = nullptr;
            bool packed_portable = false;
            if (pack_elems_max > 0) {
                packed_tmp = portable_aligned_alloc(ACC_ALIGN, pack_elems_max * sizeof(float));
                packed_portable = (packed_tmp != nullptr);
                if (!packed_tmp) {
                    packed_tmp = malloc(pack_elems_max * sizeof(float));
                    packed_portable = false;
                    if (!packed_tmp) {
                        // allocation failed -> cleanup and continue
                        if (acc_tile_buf_portable) portable_aligned_free(acc_tile_buf_tmp); else free(acc_tile_buf_tmp);
                        continue;
                    }
                }
            }

            int rb_eff = std::min(Rb, rowsA - rb);
            float* acc_tile = acc_tile_buf; // prefix usage

            // init acc_tile from denseC (copy nb_eff floats), zero pad
            for (int local_i = 0; local_i < rb_eff; ++local_i) {
                int i = rb + local_i;
                float* Crow = denseC_p + (size_t)i * (size_t)colsC + (size_t)cb;
                float* dest_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;
                if (nb_eff > 0) memcpy(dest_row, Crow, sizeof(float) * (size_t)nb_eff);
                for (int t = nb_eff; t < acc_elems_per_row; ++t) dest_row[t] = 0.0f;
            }

            // cursor initialised to row_ptr for each local row
            std::vector<MKL_INT> cursor(rb_eff);
            for (int local_i = 0; local_i < rb_eff; ++local_i) cursor[local_i] = row_ptr_p[rb + local_i];

            // kb loop
            for (int kb = 0; kb < colsA; kb += Kc) {
                int kc_eff = std::min(Kc, colsA - kb);
                if (kc_eff <= 0) continue;
                if ((size_t)kc_eff > SIZE_MAX / (size_t)nb_eff) continue;

                float* packedB = (float*)packed_tmp;
                // pack B rows [kb .. kb+kc_eff) for columns [cb .. cb+nb_eff)
                for (int kk = 0; kk < kc_eff; ++kk) {
                    const float* Brow = denseB_p + (size_t)(kb + kk) * (size_t)colsC + (size_t)cb;
                    float* dest = packedB + (size_t)kk * (size_t)nb_eff;
                    if (nb_eff > 0) memcpy(dest, Brow, sizeof(float) * (size_t)nb_eff);
                }

                // process local rows
                for (int local_i = 0; local_i < rb_eff; ++local_i) {
                    int i = rb + local_i;
                    float* acc_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;

                    MKL_INT it0 = cursor[local_i];
                    MKL_INT end = row_ptr_p[i + 1];
                    while (it0 < end && col_idx_p[it0] < kb) ++it0;
                    cursor[local_i] = it0;
                    MKL_INT it1 = it0;
                    MKL_INT kblock_end = kb + kc_eff;
                    while (it1 < end && col_idx_p[it1] < kblock_end) ++it1;
                    if (it0 >= it1) continue;

                    for (MKL_INT p = it0; p < it1; ++p) {
                        MKL_INT k_col = col_idx_p[p];
                        float v = val_p[p];
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
                    } // p
                } // local_i
            } // kb

            // write back acc_tile to C
            for (int local_i = 0; local_i < rb_eff; ++local_i) {
                int i = rb + local_i;
                float* Crow = denseC_p + (size_t)i * (size_t)colsC + (size_t)cb;
                float* src_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;
                int j = 0;
                for (; j + VEC_WIDTH - 1 < nb_eff; j += VEC_WIDTH) {
                    __m256 tmpv = _mm256_load_ps(src_row + j);    // aligned load (acc_tile is aligned)
                    _mm256_storeu_ps(Crow + j, tmpv);
                }
                for (; j < nb_eff; ++j) Crow[j] = src_row[j];
            }

            // cleanup: free buffers (use correct portable flags)
            if (packed_tmp) {
                if (packed_portable) portable_aligned_free(packed_tmp);
                else free(packed_tmp);
                packed_tmp = nullptr;
            }
            if (acc_tile_buf_tmp) {
                if (acc_tile_buf_portable) portable_aligned_free(acc_tile_buf_tmp);
                else free(acc_tile_buf_tmp);
                acc_tile_buf_tmp = nullptr;
            }
        }
    }

    return true;
}
