// optimized: acc_tile init moved outside kb loop, memcpy copy, reuse acc_tile per (cb,rb)
#include "MKL_Sparse_Methods.h"
#include "csr_builder.h"
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <thread>

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

    // 3) tiling params (unchanged)
    Tiling t = compute_tiling(rowsA, colsA, colsC, /*prefer_threads*/ 0, /*util_ratio*/0.5, L2_BYTES, VEC_WIDTH);
    int Kc = t.Kc;
    int Nb = t.Nb;
    int Rb = t.Rb;
    int num_threads = t.num_threads;

    // clamp sensible bounds (safety)
    Kc = std::max(8, std::min(Kc, colsA));
    Nb = std::max(16, std::min(Nb, colsC));
    Rb = std::max(1, std::min(Rb, rowsA));
    if (num_threads > rowsA) num_threads = rowsA;

    // persistent thread pool (static so it's reused across calls)
    static std::unique_ptr<ThreadPool> pool;
    static std::mutex pool_init_mtx;
    {
        std::lock_guard<std::mutex> lk(pool_init_mtx);
        if (!pool) pool.reset(new ThreadPool((size_t)num_threads));
    }
    //printf("Kc=%d,Nb=%d,Rb=%d,num_threads=%d\n", Kc, Nb, Rb, num_threads);

    // Phase 1: parallel count nonzeros per row -> rowCounts
    std::vector<int> rowCounts((size_t)rowsA, 0);

    int chunk = (rowsA + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);


    for (int cb = 0; cb < colsC; cb += Nb) {
        int nb_eff = std::min(Nb, colsC - cb);
        futures.push_back(pool->enqueue([=]() {
            int vec_end = (nb_eff / VEC_WIDTH) * VEC_WIDTH;
            int acc_elems_per_row = ((nb_eff + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;

            // allocate a reusable acc_tile buffer sized for maximum rb (Rb) for this cb
            size_t max_rb_eff = (size_t)std::min(Rb, rowsA);
            //if (acc_elems_per_row == 0) continue;
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

            for (int kb = 0; kb < colsA; kb += Kc) {
                int kc_eff = std::min(Kc, colsA - kb);
                MKL_INT kblock_end = kb + kc_eff;
                // iterate over rb blocks

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

                    std::vector<MKL_INT> cursor(rowsA);
                    for (int i = 0; i < rowsA; ++i) cursor[i] = row_ptr[i];
                    // ---- now for this (cb,rb) iterate over kb and accumulate directly into acc_tile ----

                    // accumulate: for each row in rb, update its acc_row using packedB
                    for (int local_i = 0; local_i < rb_eff; ++local_i) {
                        int i = rb + local_i;
                        float* acc_row = acc_tile + (size_t)local_i * (size_t)acc_elems_per_row;

                        MKL_INT start = row_ptr[i], end = row_ptr[i + 1];
                        if (start >= end) continue;
                        //MKL_INT it0 = cursor[i];
                        //MKL_INT end = row_ptr[i + 1];
                        //while (it0 < end && col_idx[it0] < kb) ++it0;
                        //cursor[i] = it0;             // 保存状态，下次从这里开始
                        //MKL_INT it1 = it0;
                        //while (it1 < end && col_idx[it1] < kblock_end) ++it1;
                        //if (it0 >= it1) continue;

                        MKL_INT* base = (MKL_INT*)col_idx.data();
                        MKL_INT* s = base + start;
                        MKL_INT* e = base + end;

                        MKL_INT* p0 = std::lower_bound(s, e, kb);
                        MKL_INT it0 = (MKL_INT)(p0 - base);

                        MKL_INT* p1 = std::lower_bound(p0, e, kblock_end);
                        MKL_INT it1 = (MKL_INT)(p1 - base);

                        //// it0/it1 within current kb block
                        //MKL_INT it0 = start;
                        //while (it0 < end && col_idx[it0] < kb) ++it0;
                        //MKL_INT it1 = it0;
                        //MKL_INT kblock_end = kb + kc_eff;
                        //while (it1 < end && col_idx[it1] < kblock_end) ++it1;
                        //if (it0 >= it1) continue;

                        MKL_INT p = it0;

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
                    // free packedB for this kb
                } // end rb loop
                if (packed_tmp) { if (packed_portable) portable_aligned_free(packed_tmp); else free(packed_tmp); }
            } // kb loop

            // free acc_tile_buf for this cb
            if (acc_tile_buf_tmp) { if (acc_tile_buf_portable) portable_aligned_free(acc_tile_buf_tmp); else free(acc_tile_buf_tmp); }
        }));
    } // cb

    for (auto& f : futures) {
        f.get();
    }

    return true;
}
