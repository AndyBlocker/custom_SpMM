#pragma once

#include <vector>
#include <mkl_spblas.h>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <atomic>

// 从 row-major 的 denseA 构造 CSR（MKL_INT 类型），并返回三个数组
// denseA: pointer to rowsA x colsA float matrix (row-major)
// 输出参数 row_ptr (size rowsA+1), col_idx (size nnz), val (size nnz)
// num_threads == 0 表示自动选择 hardware_concurrency()
bool build_csr_from_denseA(const float* denseA,
    int rowsA,
    int colsA,
    std::vector<MKL_INT>& row_ptr,
    std::vector<MKL_INT>& col_idx,
    std::vector<float>& val,
    int num_threads = 0);


class ThreadPool {
public:
    ThreadPool(size_t nthreads = 1) : stop_flag(false) {
        for (size_t i = 0; i < nthreads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(this->mtx);
                        this->cv.wait(lk, [this] { return this->stop_flag || !this->tasks.empty(); });
                        if (this->stop_flag && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
                });
        }
    }

    template<typename F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using R = decltype(f());
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task_ptr->get_future();
        {
            std::unique_lock<std::mutex> lk(mtx);
            tasks.emplace([task_ptr]() { (*task_ptr)(); });
        }
        cv.notify_one();
        return fut;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx);
            stop_flag = true;
        }
        cv.notify_all();
        for (auto& t : workers) if (t.joinable()) t.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop_flag;
};