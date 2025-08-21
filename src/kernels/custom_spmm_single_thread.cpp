// optimized: acc_tile init moved outside kb loop, memcpy copy, reuse acc_tile per (cb,rb)
#include "../include/MKL_Sparse_Methods.h"
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
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

// Moved to MKL_Sparse_Methods.cpp as the primary implementation
#if 0
bool MKL_Sparse_CooXDense_Fast_Gustavson_new_yk_single_thread(
    float* denseA,
    float* denseB,
    float* denseC,
    int rowsA,
    int colsA,
    int colsC,
    int /*flag*/)
{
    // 1) build CSR
    std::vector<int> rowCounts(rowsA, 0);
    for (int i = 0; i < rowsA; ++i) {
        int c = 0;
        float* Ai = denseA + (size_t)i * colsA;
        for (int j = 0; j < colsA; ++j) if (Ai[j] != 0.0f) ++c;
        rowCounts[i] = c;
    }
    std::vector<MKL_INT> row_ptr(rowsA + 1, 0);
    for (int i = 0; i < rowsA; ++i) row_ptr[i + 1] = row_ptr[i] + rowCounts[i];
    int nnz = row_ptr[rowsA];
    std::vector<MKL_INT> col_idx(nnz);
    std::vector<float> val(nnz);
    for (int i = 0; i < rowsA; ++i) {
        int base = row_ptr[i], off = 0;
        float* Ai = denseA + (size_t)i * colsA;
        for (int j = 0; j < colsA; ++j) {
            float x = Ai[j];
            if (x != 0.0f) { col_idx[base + off] = j; val[base + off] = x; ++off; }
        }
    }

    // 2) zero C (keeps original behavior)
    std::memset(denseC, 0, sizeof(float) * (size_t)rowsA * (size_t)colsC);

    // 3) tiling params (unchanged)
    const double util_ratio = 0.5;
    const int target_bytes = static_cast<int>(L2_BYTES * util_ratio);
    int Kc = 256, Nb = std::min(colsC, 128);
    int Rb = std::max(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    if (Rb > rowsA) Rb = rowsA;
    if (Rb < 1) {
        Kc = std::max<int>(8, target_bytes / (4 * std::max(1, Nb)) - 1);
        Rb = std::max<int>(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    }
    Kc = std::max(8, std::min(Kc, colsA));
    Nb = std::max(16, std::min(Nb, colsC));
    Rb = std::max(1, std::min(Rb, rowsA));

    for (int cb = 0; cb < colsC; cb += Nb) {
        int nb_eff = std::min(Nb, colsC - cb);
        int vec_end = (nb_eff / VEC_WIDTH) * VEC_WIDTH;
        int acc_elems_per_row = ((nb_eff + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;

        // allocate a reusable acc_tile buffer sized for maximum rb (Rb) for this cb
        size_t max_rb_eff = (size_t)std::min(Rb, rowsA);
        if (acc_elems_per_row == 0) continue;
        //if (max_rb_eff > (SIZE_MAX / acc_elems_per_row)) { fprintf(stderr, "acc_tile overflow\n"); return false; }
        size_t acc_tile_max_elems = max_rb_eff * (size_t)acc_elems_per_row;
        void* acc_tile_buf_tmp = portable_aligned_alloc(ACC_ALIGN, acc_tile_max_elems * sizeof(float));
        bool acc_tile_buf_portable = true;
        if (!acc_tile_buf_tmp) {
            acc_tile_buf_tmp = malloc(acc_tile_max_elems * sizeof(float));
            acc_tile_buf_portable = false;
            //if (!acc_tile_buf_tmp) { fprintf(stderr, "acc_tile_buf alloc fail\n"); return false; }
        }
        float* acc_tile_buf = (float*)acc_tile_buf_tmp;

        // iterate over rb blocks
        for (int rb = 0; rb < rowsA; rb += Rb) {
            int rb_eff = std::min(Rb, rowsA - rb);
            if (rb_eff <= 0) continue;

            // acc_tile for this rb is the prefix of acc_tile_buf
            float* acc_tile = acc_tile_buf; // size = rb_eff * acc_elems_per_row floats

            // ---- initialization: COPY denseC small block into acc_tile (only once per (cb,rb)) ----
            // Use memcpy per row (fast) and only write small padding to zero per row if needed.
            for (int local_i = 0; local_i < rb_eff; ++local_i) {
                int i = rb + local_i;
                float* Crow = denseC + (size_t)i * (size_t)colsC + (size_t)cb;
                float* dest_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;
                // copy nb_eff floats
                if (nb_eff > 0) memcpy(dest_row, Crow, sizeof(float) * (size_t)nb_eff);
                // zero padding up to acc_elems_per_row (small <= VEC_WIDTH)
                for (int t = nb_eff; t < acc_elems_per_row; ++t) dest_row[t] = 0.0f;
            }

            // ---- now for this (cb,rb) iterate over kb and accumulate directly into acc_tile ----
            for (int kb = 0; kb < colsA; kb += Kc) {
                int kc_eff = std::min(Kc, colsA - kb);

                // pack denseB for this (kb,cb)
                size_t pack_elems = (size_t)kc_eff * (size_t)nb_eff;
                void* packed_tmp = portable_aligned_alloc(ACC_ALIGN, pack_elems * sizeof(float));
                bool packed_portable = true;
                if (!packed_tmp) {
                    packed_tmp = malloc(pack_elems * sizeof(float));
                    packed_portable = false;
                    //if (!packed_tmp) {
                    //    fprintf(stderr, "packedB alloc failed\n");
                    //    if (acc_tile_buf_tmp) { if (acc_tile_buf_portable) portable_aligned_free(acc_tile_buf_tmp); else free(acc_tile_buf_tmp); }
                    //    return false;
                    //}
                }
                float* packedB = (float*)packed_tmp;
                for (int kk = 0; kk < kc_eff; ++kk) {
                    const float* Brow = denseB + (size_t)(kb + kk) * (size_t)colsC + (size_t)cb;
                    float* dest = packedB + (size_t)kk * (size_t)nb_eff;
                    // memcpy contiguous nb_eff floats
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
                        int step = VEC_WIDTH * 4;
                        for (; j + step - VEC_WIDTH < vec_end; j += step) {
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

            // ---- write back acc_tile once for (cb,rb) ----
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
    } // cb

    return true;
}
#endif
