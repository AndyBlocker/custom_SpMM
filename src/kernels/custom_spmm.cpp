// custom_spmm_avx2_rowtasks_reuse_scratch.cpp
// AVX2 implementation optimized for Linux (reuse scratch, no row-locks) and Windows-compatible
#include "../include/MKL_Sparse_Methods.h"
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <mkl_spblas.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <memory>

// portable aligned alloc/free (header stores original pointer) - cross-platform
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

// -------------------- CSR 构造阶段 --------------------
struct CSRCountArgs1 { int tid, nt, rowsA, colsA; float* denseA; std::vector<int>* rowCounts; };
static void csr_count_worker(CSRCountArgs1* a) {
    int tid = a->tid, nt = a->nt;
    int rowsA = a->rowsA, colsA = a->colsA;
    float* A = a->denseA;
    auto& counts = *a->rowCounts;
    int chunk = (rowsA + nt - 1) / nt;
    int r0 = tid * chunk;
    int r1 = std::min(r0 + chunk, rowsA);
    for (int i = r0; i < r1; ++i) {
        int c = 0;
        float* Ai = A + (size_t)i * colsA;
        for (int j = 0; j < colsA; ++j)
            if (Ai[j] != 0.0f) ++c;
        counts[i] = c;
    }
}

struct CSRFillArgs1 {
    int tid, nt, rowsA, colsA;
    float* denseA;
    const std::vector<MKL_INT>* row_ptr;
    std::vector<int>* localOffsets;
    std::vector<MKL_INT>* col_idx;
    std::vector<float>* val;
};
static void csr_fill_worker(CSRFillArgs1* a) {
    int tid = a->tid, nt = a->nt;
    int rowsA = a->rowsA, colsA = a->colsA;
    float* A = a->denseA;
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
        float* Ai = A + (size_t)i * colsA;
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
}

// -------------------- Tasks --------------------
struct RowTask {
    int row_start; // inclusive
    int row_end;   // exclusive
};

// -------------------- Job --------------------
struct Job {
    float* denseB;
    float* denseC;
    const std::vector<MKL_INT>* row_ptr;
    const std::vector<MKL_INT>* col_idx;
    const std::vector<float>* val;
    int rowsA, colsA, colsC;
    int Kc, Nb, Rb;
    std::vector<RowTask> tasks;
};

// -------------------- ThreadPool --------------------
struct ThreadPool {
    std::vector<std::thread> workers;
    int nworkers = 0;

    std::mutex job_mutex;
    std::condition_variable job_cv;
    std::condition_variable job_done_cv;

    std::atomic<int> next_task{0};
    std::atomic<int> outstanding{0};

    Job current_job;
    bool has_job = false;
    bool shutting_down = false;

    // per-worker persistent scratch buffers
    float** worker_scratch = nullptr;
    size_t* worker_scratch_elems = nullptr;
    MKL_INT** worker_row_kbegin = nullptr;
    MKL_INT** worker_row_kend = nullptr;
    size_t* worker_row_k_elems = nullptr;

    static constexpr int PREFETCH_P = 4;

    ThreadPool() = default;

    // worker entry
    void worker_loop(int wid) {
        while (true) {
            std::unique_lock<std::mutex> lk(job_mutex);
            job_cv.wait(lk, [this]() { return has_job || shutting_down; });
            if (shutting_down) { lk.unlock(); break; }
            Job* jobp = &current_job;
            lk.unlock();

            const std::vector<RowTask>& tasks = jobp->tasks;
            const MKL_INT* row_ptr = jobp->row_ptr->data();
            const MKL_INT* col_idx = jobp->col_idx->data();
            const float* val = jobp->val->data();
            float* denseB = jobp->denseB;
            float* denseC = jobp->denseC;
            int colsA = jobp->colsA;
            int colsC = jobp->colsC;
            int Kc = jobp->Kc;
            int Nb = jobp->Nb;
            int Rb = jobp->Rb;

            int ntasks = (int)tasks.size();
            while (true) {
                int tidx = next_task.fetch_add(1, std::memory_order_relaxed);
                if (tidx >= ntasks) break;

                const RowTask &task = tasks[tidx];
                int r0 = task.row_start;
                int r1 = task.row_end;
                int rows_in_task = r1 - r0;
                if (rows_in_task <= 0) {
                    int rem = outstanding.fetch_sub(1, std::memory_order_acq_rel) - 1;
                    if (rem == 0) {
                        std::unique_lock<std::mutex> lk2(job_mutex);
                        job_done_cv.notify_one();
                    }
                    continue;
                }

                // Precompute nb_eff/vec_end etc per column block
                for (int cb = 0; cb < colsC; cb += Nb) {
                    int nb_eff = std::min(Nb, colsC - cb);
                    int vec_end = (nb_eff / VEC_WIDTH) * VEC_WIDTH;
                    int acc_elems_per_row = ((nb_eff + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;
                    size_t acc_count = (size_t)rows_in_task * (size_t)acc_elems_per_row;

                    // Use per-worker scratch if available and big enough
                    float* acc_buf = nullptr;
                    if (worker_scratch && worker_scratch[wid] && worker_scratch_elems[wid] >= acc_count) {
                        acc_buf = worker_scratch[wid];
                        // zero only used region
                        memset(acc_buf, 0, acc_count * sizeof(float));
                    } else {
                        // fallback allocate aligned temporary (rare)
                        void* tmp = portable_aligned_alloc(ACC_ALIGN, acc_count * sizeof(float));
                        if (!tmp) tmp = malloc(acc_count * sizeof(float));
                        acc_buf = (float*)tmp;
                        if (acc_buf) memset(acc_buf, 0, acc_count * sizeof(float));
                    }

                    // Use per-worker row_k buffers (they are sized to max_rows_in_task by submit_and_wait)
                    MKL_INT* row_kbegin = nullptr;
                    MKL_INT* row_kend = nullptr;
                    if (worker_row_kbegin && worker_row_kbegin[wid] && worker_row_k_elems[wid] >= (size_t)rows_in_task) {
                        row_kbegin = worker_row_kbegin[wid];
                        row_kend = worker_row_kend[wid];
                    } else {
                        row_kbegin = (MKL_INT*)malloc(sizeof(MKL_INT) * (size_t)rows_in_task);
                        row_kend   = (MKL_INT*)malloc(sizeof(MKL_INT) * (size_t)rows_in_task);
                    }

                    // init per-row cursors
                    for (int ii = 0; ii < rows_in_task; ++ii) {
                        int i = r0 + ii;
                        MKL_INT rbeg = row_ptr[i];
                        row_kbegin[ii] = rbeg;
                        row_kend[ii] = rbeg;
                    }

                    // main blocked loops: iterate kb blocks
                    for (int rb = r0; rb < r1; rb += Rb) {
                        int rb_eff = std::min(Rb, r1 - rb);

                        for (int kb = 0; kb < colsA; kb += Kc) {
                            int kc_eff = std::min(Kc, colsA - kb);

                            for (int local_i = 0; local_i < rb_eff; ++local_i) {
                                int i = rb + local_i;
                                int off = i - r0;

                                MKL_INT rstart = row_ptr[i];
                                MKL_INT rend = row_ptr[i+1];
                                if (rstart >= rend) continue;

                                MKL_INT it0 = row_kbegin[off];
                                while (it0 < rend && col_idx[it0] < (MKL_INT)kb) ++it0;

                                MKL_INT it1 = row_kend[off];
                                if (it1 < it0) it1 = it0;
                                const MKL_INT k_end = (MKL_INT)(kb + kc_eff);
                                while (it1 < rend && col_idx[it1] < k_end) ++it1;

                                row_kbegin[off] = it0;
                                row_kend[off] = it1;

                                if (it0 >= it1) continue;

                                float* acc_row = acc_buf + (size_t)off * (size_t)acc_elems_per_row;

                                // p outer, j inner
                                // unroll-4 version
                                MKL_INT p = it0;
                                // handle groups of 4
                                for (; p + 3 < it1; p += 4) {
                                    // prefetch future col/val entries
                                    if (p + PREFETCH_P < it1) {
                                        prefetch_read(&col_idx[p + PREFETCH_P]);
                                        prefetch_read(&val[p + PREFETCH_P]);
                                        // also prefetch a bit further for the p+1..p+3 entries (optional)
                                        if (p + 1 + PREFETCH_P < it1) { prefetch_read(&col_idx[p + 1 + PREFETCH_P]); prefetch_read(&val[p + 1 + PREFETCH_P]); }
                                    }

                                    MKL_INT k0 = col_idx[p + 0]; float v0 = val[p + 0];
                                    MKL_INT k1 = col_idx[p + 1]; float v1 = val[p + 1];
                                    MKL_INT k2 = col_idx[p + 2]; float v2 = val[p + 2];
                                    MKL_INT k3 = col_idx[p + 3]; float v3 = val[p + 3];

                                    const float* B0 = denseB + (size_t)k0 * (size_t)colsC + (size_t)cb;
                                    const float* B1 = denseB + (size_t)k1 * (size_t)colsC + (size_t)cb;
                                    const float* B2 = denseB + (size_t)k2 * (size_t)colsC + (size_t)cb;
                                    const float* B3 = denseB + (size_t)k3 * (size_t)colsC + (size_t)cb;

                                    // prefetch B rows (helps if memory latency dominates)
                                    prefetch_read(B0);
                                    prefetch_read(B1);
                                    prefetch_read(B2);
                                    prefetch_read(B3);

                                    __m256 v0v = _mm256_set1_ps(v0);
                                    __m256 v1v = _mm256_set1_ps(v1);
                                    __m256 v2v = _mm256_set1_ps(v2);
                                    __m256 v3v = _mm256_set1_ps(v3);

                                    int j = 0;
                                    // vectorized part: load acc once per j, load 4 B vectors, do 4 fmadds, store once
                                    for (; j + VEC_WIDTH - 1 < vec_end; j += VEC_WIDTH) {
                                        __m256 accv = _mm256_loadu_ps(acc_row + j);      // 若 acc_row 对齐且你确认可用 _mm256_load_ps，可替换
                                        __m256 b0 = _mm256_loadu_ps(B0 + j);             // 若 B0 已 pack 且对齐，可替换为 _mm256_load_ps
                                        __m256 b1 = _mm256_loadu_ps(B1 + j);
                                        __m256 b2 = _mm256_loadu_ps(B2 + j);
                                        __m256 b3 = _mm256_loadu_ps(B3 + j);

                                        accv = _mm256_fmadd_ps(v0v, b0, accv);
                                        accv = _mm256_fmadd_ps(v1v, b1, accv);
                                        accv = _mm256_fmadd_ps(v2v, b2, accv);
                                        accv = _mm256_fmadd_ps(v3v, b3, accv);

                                        _mm256_storeu_ps(acc_row + j, accv);             // 同上：若对齐，可替换为 _mm256_store_ps
                                    }

                                    // scalar tail for jj in [vec_end, nb_eff)
                                    if (vec_end < nb_eff) {
                                        for (int jj = vec_end; jj < nb_eff; ++jj) {
                                            acc_row[jj] += v0 * B0[jj] + v1 * B1[jj] + v2 * B2[jj] + v3 * B3[jj];
                                        }
                                    }
                                } // end groups of 4

                                // handle remaining p's (0..3 leftovers)
                                for (; p < it1; ++p) {
                                    if (p + PREFETCH_P < it1) {
                                        prefetch_read(&col_idx[p + PREFETCH_P]);
                                        prefetch_read(&val[p + PREFETCH_P]);
                                    }
                                    MKL_INT k_col = col_idx[p];
                                    float v = val[p];
                                    const float* Brow = denseB + (size_t)k_col * (size_t)colsC + (size_t)cb;
                                    prefetch_read(Brow);

                                    __m256 v_vec = _mm256_set1_ps(v);
                                    int j = 0;
                                    for (; j + VEC_WIDTH - 1 < vec_end; j += VEC_WIDTH) {
                                        __m256 bvec = _mm256_loadu_ps(Brow + j);
                                        __m256 accv = _mm256_loadu_ps(acc_row + j);
                                        accv = _mm256_fmadd_ps(v_vec, bvec, accv);
                                        _mm256_storeu_ps(acc_row + j, accv);
                                    }
                                    if (vec_end < nb_eff) {
                                        for (int jj = vec_end; jj < nb_eff; ++jj) {
                                            acc_row[jj] += v * Brow[jj];
                                        }
                                    }
                                }
                            } // local_i
                        } // rb
                    } // kb

                    // write back acc_buf into C (no locks needed — each row belongs to exactly one task)
                    for (int ii = 0; ii < rows_in_task; ++ii) {
                        int i = r0 + ii;
                        float* Crow = denseC + (size_t)i * (size_t)colsC + (size_t)cb;
                        prefetch_read(Crow);
                        int j = 0;
                        for (; j < vec_end; j += VEC_WIDTH) {
                            __m256 cvec = _mm256_loadu_ps(Crow + j);
                            __m256 accv = _mm256_loadu_ps(acc_buf + (size_t)ii * (size_t)acc_elems_per_row + j);
                            cvec = _mm256_add_ps(cvec, accv);
                            _mm256_storeu_ps(Crow + j, cvec);
                        }
                        if (vec_end < nb_eff) {
                            for (int jj = vec_end; jj < nb_eff; ++jj) {
                                Crow[jj] += acc_buf[(size_t)ii * (size_t)acc_elems_per_row + jj];
                            }
                        }
                    }

                    // free temporary row_k arrays if we allocated them here
                    if (!(worker_row_kbegin && worker_row_kbegin[wid] && worker_row_k_elems[wid] >= (size_t)rows_in_task)) {
                        if (row_kbegin) free(row_kbegin);
                        if (row_kend) free(row_kend);
                    }

                    // if acc_buf wasn't worker scratch, free it
                    if (!(worker_scratch && worker_scratch[wid] && worker_scratch_elems[wid] >= acc_count)) {
                        if (acc_buf) {
                            portable_aligned_free(acc_buf); // portable_aligned_free accepts nullptr safely
                        }
                    }
                } // cb

                int rem = outstanding.fetch_sub(1, std::memory_order_acq_rel) - 1;
                if (rem == 0) {
                    std::unique_lock<std::mutex> lk2(job_mutex);
                    job_done_cv.notify_one();
                }
            } // tasks loop
        } // while true
    } // worker_loop

    bool init(int nthreads) {
        if (nthreads <= 0) return false;
        nworkers = nthreads;
        try {
            workers.reserve(nworkers);
            for (int i = 0; i < nworkers; ++i) {
                workers.emplace_back(&ThreadPool::worker_loop, this, i);
            }
        } catch (...) {
            for (auto &t : workers) if (t.joinable()) t.join();
            workers.clear();
            nworkers = 0;
            return false;
        }

        // allocate arrays for per-worker pointers (actual buffers allocated in submit_and_wait)
        worker_scratch = (float**)malloc(sizeof(float*) * (size_t)nworkers);
        worker_scratch_elems = (size_t*)malloc(sizeof(size_t) * (size_t)nworkers);
        worker_row_kbegin = (MKL_INT**)malloc(sizeof(MKL_INT*) * (size_t)nworkers);
        worker_row_kend   = (MKL_INT**)malloc(sizeof(MKL_INT*) * (size_t)nworkers);
        worker_row_k_elems = (size_t*)malloc(sizeof(size_t) * (size_t)nworkers);
        if (!worker_scratch || !worker_scratch_elems || !worker_row_kbegin || !worker_row_kend || !worker_row_k_elems) {
            if (worker_scratch) free(worker_scratch);
            if (worker_scratch_elems) free(worker_scratch_elems);
            if (worker_row_kbegin) free(worker_row_kbegin);
            if (worker_row_kend) free(worker_row_kend);
            if (worker_row_k_elems) free(worker_row_k_elems);
            worker_scratch = nullptr; worker_scratch_elems = nullptr;
            worker_row_kbegin = nullptr; worker_row_kend = nullptr; worker_row_k_elems = nullptr;
            return false;
        }
        for (int i = 0; i < nworkers; ++i) {
            worker_scratch[i] = nullptr; worker_scratch_elems[i] = 0;
            worker_row_kbegin[i] = nullptr; worker_row_kend[i] = nullptr; worker_row_k_elems[i] = 0;
        }
        return true;
    }

    // prepare (or grow) per-worker scratch to hold elems_per_worker floats and row_k buffers sized max_rows_in_task
    void ensure_worker_scratch_and_rows(size_t elems_per_worker, size_t max_rows_in_task) {
        for (int i = 0; i < nworkers; ++i) {
            // scratch buffer
            if (worker_scratch[i] && worker_scratch_elems[i] < elems_per_worker) {
                portable_aligned_free(worker_scratch[i]);
                worker_scratch[i] = nullptr; worker_scratch_elems[i] = 0;
            }
            if (worker_scratch[i] == nullptr && elems_per_worker > 0) {
                void* ptr = portable_aligned_alloc(ACC_ALIGN, elems_per_worker * sizeof(float));
                if (!ptr) ptr = malloc(elems_per_worker * sizeof(float));
                if (ptr) { worker_scratch[i] = (float*)ptr; worker_scratch_elems[i] = elems_per_worker; }
                else { worker_scratch[i] = nullptr; worker_scratch_elems[i] = 0; }
            }
            // row_k buffers
            if (worker_row_kbegin[i] != nullptr && worker_row_k_elems[i] < max_rows_in_task) {
                free(worker_row_kbegin[i]); worker_row_kbegin[i] = nullptr;
            }
            if (worker_row_kend[i] != nullptr && worker_row_k_elems[i] < max_rows_in_task) {
                free(worker_row_kend[i]); worker_row_kend[i] = nullptr;
            }
            if (worker_row_kbegin[i] == nullptr && max_rows_in_task > 0) {
                worker_row_kbegin[i] = (MKL_INT*)malloc(sizeof(MKL_INT) * max_rows_in_task);
                worker_row_kend[i]   = (MKL_INT*)malloc(sizeof(MKL_INT) * max_rows_in_task);
                if (worker_row_kbegin[i] && worker_row_kend[i]) worker_row_k_elems[i] = max_rows_in_task;
                else {
                    if (worker_row_kbegin[i]) { free(worker_row_kbegin[i]); worker_row_kbegin[i] = nullptr; }
                    if (worker_row_kend[i]) { free(worker_row_kend[i]); worker_row_kend[i] = nullptr; }
                    worker_row_k_elems[i] = 0;
                }
            } else {
                worker_row_k_elems[i] = max_rows_in_task;
            }
        }
    }

    void submit_and_wait(Job&& job) {
        // compute max_rows_in_task and elems_per_worker based on job.tasks
        size_t max_rows_in_task = 0;
        for (const RowTask &t : job.tasks) {
            size_t r = (size_t)(t.row_end - t.row_start);
            if (r > max_rows_in_task) max_rows_in_task = r;
        }
        if (max_rows_in_task == 0) max_rows_in_task = 1;

        int nb_worst = job.Nb;
        int acc_elems_per_row_worst = ((nb_worst + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;
        size_t elems_per_worker = max_rows_in_task * (size_t)acc_elems_per_row_worst;

        // ensure worker scratch allocated and sized
        ensure_worker_scratch_and_rows(elems_per_worker, max_rows_in_task);

        {
            std::unique_lock<std::mutex> lk(job_mutex);
            current_job = std::move(job);
            next_task.store(0, std::memory_order_relaxed);
            outstanding.store((int)current_job.tasks.size(), std::memory_order_relaxed);
            has_job = true;
            job_cv.notify_all();

            while (outstanding.load(std::memory_order_acquire) > 0) {
                job_done_cv.wait(lk);
            }

            has_job = false;
            current_job.tasks.clear();
        }
    }

    void destroy() {
        {
            std::unique_lock<std::mutex> lk(job_mutex);
            shutting_down = true;
            job_cv.notify_all();
        }

        for (auto &t : workers) if (t.joinable()) t.join();
        workers.clear();

        if (worker_scratch) {
            for (int i = 0; i < nworkers; ++i) if (worker_scratch[i]) portable_aligned_free(worker_scratch[i]);
            free(worker_scratch); worker_scratch = nullptr;
        }
        if (worker_scratch_elems) { free(worker_scratch_elems); worker_scratch_elems = nullptr; }
        if (worker_row_kbegin) {
            for (int i = 0; i < nworkers; ++i) if (worker_row_kbegin[i]) free(worker_row_kbegin[i]);
            free(worker_row_kbegin); worker_row_kbegin = nullptr;
        }
        if (worker_row_kend) {
            for (int i = 0; i < nworkers; ++i) if (worker_row_kend[i]) free(worker_row_kend[i]);
            free(worker_row_kend); worker_row_kend = nullptr;
        }
        if (worker_row_k_elems) { free(worker_row_k_elems); worker_row_k_elems = nullptr; }

        nworkers = 0;
    }
};

static ThreadPool* g_pool = nullptr;
static std::mutex g_pool_init_mutex;

static int find_row_by_nnz(const std::vector<MKL_INT>& row_ptr, int rowsA, MKL_INT idx) {
    int lo = 0, hi = rowsA - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (row_ptr[mid + 1] <= idx) lo = mid + 1;
        else if (row_ptr[mid] > idx) hi = mid - 1;
        else return mid;
    }
    if (lo >= rowsA) return rowsA - 1;
    return lo;
}

// choose threads & target (unchanged)
static void choose_threads_and_target(int rowsA, int colsA, int colsC, int nnz,
                                      int &out_hw_threads, int &out_target_nnz_per_task) {
    unsigned int logical_cpus_u = std::thread::hardware_concurrency();
    int logical_cpus = (int)(logical_cpus_u ? logical_cpus_u : 1);
    double total = (double)rowsA * (double)colsA;
    double density = (total > 0.0) ? ((double)nnz / total) : 0.0;

    double f = 0.125;
    if (density >= 0.10) f = 0.5;
    else if (density >= 0.05) f = 0.5;
    else if (density >= 0.02) f = 0.5;

    int hw_threads = std::max(1, (int)std::round(logical_cpus * f));
    hw_threads = std::min(logical_cpus, hw_threads);
    if (nnz < hw_threads) hw_threads = std::max(1, std::min(nnz, logical_cpus));

    int base_factor = (density >= 0.10) ? 8 : (density >= 0.02) ? 12 : (density >= 0.002) ? 16 : 24;
    long long approx_tasks = (long long)hw_threads * (long long)base_factor;
    if (approx_tasks < 1) approx_tasks = 1;
    int target_nnz_per_task = std::max(32, (int)std::max(1LL, (long long)nnz / approx_tasks));
    if (target_nnz_per_task > nnz) target_nnz_per_task = std::max(1024, nnz);

    out_hw_threads = hw_threads;
    out_target_nnz_per_task = target_nnz_per_task;
}

// Main exposed function: now create tasks by grouping whole rows (no row overlaps)
bool MKL_Sparse_CooXDense_Fast_Gustavson_new(
    float *denseA,
    float *denseB,
    float *denseC,
    int rowsA,
    int colsA,
    int colsC,
    int /*flag*/)
{
    //auto t_start = std::chrono::steady_clock::now();
    int nt = 16;
    if (rowsA <= 64) nt = 1; else if (rowsA <= 1024) nt = 4;
    if (nt < 1) nt = 4;

    // count nonzeros per row
    std::vector<int> rowCounts(rowsA, 0);
    std::vector<std::thread> ct;
    ct.reserve(nt);
    std::vector<CSRCountArgs1> cargs(nt);
    for (int t = 0; t < nt; ++t) {
        cargs[t] = {t, nt, rowsA, colsA, denseA, &rowCounts};
        ct.emplace_back(csr_count_worker, &cargs[t]);
    }
    for (auto &th : ct) if (th.joinable()) th.join();

    // build CSR
    std::vector<MKL_INT> row_ptr(rowsA + 1, 0);
    for (int i = 0; i < rowsA; ++i) row_ptr[i+1] = row_ptr[i] + rowCounts[i];
    int nnz = row_ptr[rowsA];

    std::vector<MKL_INT> col_idx(nnz);
    std::vector<float> val(nnz);
    std::vector<int> localOffsets(rowsA, 0);
    std::vector<std::thread> ft;
    ft.reserve(nt);
    std::vector<CSRFillArgs1> fargs(nt);
    for (int t = 0; t < nt; ++t) {
        fargs[t] = {t, nt, rowsA, colsA, denseA, &row_ptr, &localOffsets, &col_idx, &val};
        ft.emplace_back(csr_fill_worker, &fargs[t]);
    }
    for (auto &th : ft) if (th.joinable()) th.join();

    //std::chrono::duration<double> diff = t_end - t_start;
    // printf("Sparse×Dense (fast): %.6f s\n",
    //        diff.count());
    // zero C
    std::memset(denseC, 0, sizeof(float) * (size_t)rowsA * (size_t)colsC);

    if (nnz == 0) return true;

    const double util_ratio = 0.5;
    const int target_bytes = static_cast<int>(L2_BYTES * util_ratio);
    int Kc = 256;
    int Nb = std::min(colsC, 256);
    int Rb = std::max(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    if (Rb > rowsA) Rb = rowsA;
    if (Rb < 1) {
        Kc = std::max<int>(8, target_bytes / (4 * std::max(1, Nb)) - 1);
        Rb = std::max<int>(1, static_cast<int>(target_bytes / (4 * std::max(1, Nb))) - Kc);
    }
    Kc = std::max(8, std::min(Kc, colsA));
    Nb = std::max(16, std::min(Nb, colsC));
    Rb = std::max(1, std::min(Rb, rowsA));

    // Build tasks: group contiguous rows so each row belongs to exactly one task.
    int hw_threads = 0, target_nnz_per_task = 0;
    choose_threads_and_target(rowsA, colsA, colsC, nnz, hw_threads, target_nnz_per_task);

    unsigned int logical_cpus_u = std::thread::hardware_concurrency();
    int logical_cpus = (int)(logical_cpus_u ? logical_cpus_u : 1);
    if (hw_threads <= 0) hw_threads = std::min(logical_cpus, nnz);
    hw_threads = std::min(hw_threads, logical_cpus);
    if (nnz < hw_threads) hw_threads = std::max(1, std::min(nnz, logical_cpus));

    if (target_nnz_per_task <= 0) target_nnz_per_task = std::max(1, nnz / (hw_threads * 8));

    std::vector<RowTask> tasks;
    tasks.reserve(std::max(1, hw_threads * 4));

    // Greedy grouping rows to approximate target_nnz_per_task
    int rstart = 0;
    int acc = 0;
    for (int i = 0; i < rowsA; ++i) {
        int rnz = rowCounts[i];
        if (acc == 0) { rstart = i; acc = rnz; }
        else acc += rnz;
        // If accumulated >= target and at least one row, cut
        if (acc >= target_nnz_per_task && i >= rstart) {
            tasks.push_back({rstart, i + 1});
            acc = 0;
        }
    }
    if (acc > 0) {
        tasks.push_back({rstart, rowsA});
    }
    if (tasks.empty()) tasks.push_back({0, rowsA});

    // Prepare job
    Job job;
    job.denseB = denseB; job.denseC = denseC;
    job.row_ptr = &row_ptr; job.col_idx = &col_idx; job.val = &val;
    job.rowsA = rowsA; job.colsA = colsA; job.colsC = colsC;
    job.Kc = Kc; job.Nb = Nb; job.Rb = Rb;
    job.tasks = std::move(tasks);

    int actual_threads = std::min(hw_threads, (int)job.tasks.size());
    if (actual_threads < 1) actual_threads = 1;

    //printf("SpMM: rowsA=%d, colsA=%d, colsC=%d, nnz=%d, hw_threads=%d, target_nnz_per_task=%d, task_num=%ld\n",
    //    rowsA, colsA, colsC, nnz, actual_threads, target_nnz_per_task, job.tasks.size());

    // init pool if needed
    std::lock_guard<std::mutex> guard(g_pool_init_mutex);
    if (!g_pool) {
        g_pool = new ThreadPool();
        if (!g_pool->init(actual_threads)) {
            fprintf(stderr, "ThreadPool init failed, fallback single-thread\n");
            delete g_pool; g_pool = nullptr;
            // fallback single-thread execution
            // build minimal Job->call local process_job_singlethread-like logic
            // but for brevity call single-thread path implemented below:
            // (reuse simplified single-thread variant)
            // Implement single-thread processing here (copy of inner loops)
            // For correctness and brevity, call an existing single-thread processing wrapper
            // But original code had process_job_singlethread; we'll implement a simple call:
            {
                // Create small wrapper job for single-thread processing below
                // We'll reuse worker code by launching a temporary ThreadPool with 1 thread
                ThreadPool tmp_pool;
                if (!tmp_pool.init(1)) return false;
                tmp_pool.submit_and_wait(std::move(job));
                tmp_pool.destroy();
            }
            return true;
        }
    }

    // Submit and wait
    g_pool->submit_and_wait(std::move(job));
    return true;
}
