#include "csr_builder.h"

// ------------------ 简单 ThreadPool（线程常驻，返回 future） ------------------


// ------------------ CSR 构造函数 ------------------
bool build_csr_from_denseA(const float* denseA,
    int rowsA,
    int colsA,
    std::vector<MKL_INT>& row_ptr,
    std::vector<MKL_INT>& col_idx,
    std::vector<float>& val,
    int num_threads)
{
    if (!denseA || rowsA <= 0 || colsA <= 0) {
        row_ptr.clear(); col_idx.clear(); val.clear();
        return true;
    }

    // choose thread count
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (num_threads <= 0) num_threads = (int)hw;
    if (num_threads > rowsA) num_threads = rowsA; // 不要超过行数

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

    for (int t = 0; t < num_threads; ++t) {
        int rs = t * chunk;
        int re = std::min(rs + chunk, rowsA);
        if (rs >= re) break;
        futures.push_back(pool->enqueue([denseA, rowsA, colsA, rs, re, &rowCounts]() {
            for (int i = rs; i < re; ++i) {
                const float* Ai = denseA + (size_t)i * (size_t)colsA;
                int cnt = 0;
                // simple scalar loop - branch predictable enough; could replace with SIMD if desired
                for (int j = 0; j < colsA; ++j) {
                    if (Ai[j] != 0.0f) ++cnt;
                }
                rowCounts[i] = cnt;
            }
            }));
    }
    for (auto& f : futures) f.get();
    futures.clear();

    // Phase 2: serial prefix-sum to build global row_ptr
    row_ptr.assign((size_t)rowsA + 1, 0);
    for (int i = 0; i < rowsA; ++i) row_ptr[i + 1] = row_ptr[i] + (MKL_INT)rowCounts[i];
    MKL_INT nnz = row_ptr[rowsA];

    // allocate global arrays
    col_idx.assign((size_t)nnz, 0);
    val.assign((size_t)nnz, 0.0f);

    if (nnz == 0) return true;

    // Phase 3: parallel fill col_idx & val directly into global arrays
    // Each thread fills rows [rs,re) and writes to non-overlapping ranges row_ptr[i]..row_ptr[i+1]-1
    for (int t = 0; t < num_threads; ++t) {
        int rs = t * chunk;
        int re = std::min(rs + chunk, rowsA);
        if (rs >= re) break;
        futures.push_back(pool->enqueue([denseA, colsA, rs, re, &row_ptr, &col_idx, &val]() {
            for (int i = rs; i < re; ++i) {
                const float* Ai = denseA + (size_t)i * (size_t)colsA;
                MKL_INT pos = row_ptr[i]; // global write position for this row (unique)
                // write sequentially starting at pos
                for (int j = 0; j < colsA; ++j) {
                    float x = Ai[j];
                    if (x != 0.0f) {
                        // pos is unique to this row and thread, safe to increment
                        col_idx[(size_t)pos] = (MKL_INT)j;
                        val[(size_t)pos] = x;
                        ++pos;
                    }
                }
            }
            }));
    }
    for (auto& f : futures) f.get();
    futures.clear();

    return true;
}
