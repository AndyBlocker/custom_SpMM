#include "MKL_Sparse_Methods.h"
#include <thread>


Tiling compute_tiling(int rowsA, int colsA, int colsC,
    int prefer_threads = 0,
    double util_ratio = 0.5,
    int L2_bytes = L2_BYTES,
    int vec_w = VEC_WIDTH)
{
    const int target_bytes = static_cast<int>(L2_bytes * util_ratio);
    const int Nb_max_init = std::min(colsC, 128);
    const int Nb_min = 16;
    const int Kc_max_init = std::min(colsA, 384);
    const int Kc_min = 8;

    // thread count resolution
    int num_threads = prefer_threads;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (num_threads <= 0) num_threads = (int)hw;
    if (num_threads > rowsA) num_threads = rowsA;

    auto align_vec_down = [&](int x)->int {
        if (x < Nb_min) return Nb_min;
        int y = (x / vec_w) * vec_w;
        if (y < Nb_min) y = Nb_min;
        return y;
        };
    auto align_vec_up = [&](int x)->int {
        if (x < Nb_min) return Nb_min;
        int y = ((x + vec_w - 1) / vec_w) * vec_w;
        if (y < Nb_min) y = Nb_min;
        return y;
        };

    auto estimate_Rb_for = [&](int Kc, int Nb)->int {
        // total bytes estimate: 4*(Kc*Nb + Rb*(Nb + Kc))
        long long KB = (long long)Kc * (long long)Nb;
        long long denom = (long long)(Nb + Kc);
        long long remain = (long long)target_bytes - 4LL * KB;
        if (remain <= 0 || denom <= 0) return 0;
        long long Rb_max = remain / (4LL * denom);
        if (Rb_max < 0) Rb_max = 0;
        if (Rb_max > rowsA) Rb_max = rowsA;
        return (int)Rb_max;
        };

    int chosen_Nb = align_vec_down(Nb_max_init);
    int chosen_Kc = std::min(Kc_max_init, colsA);
    int chosen_Rb = 1;
    bool found = false;

    // search decreasing Nb
    for (int trial_Nb = chosen_Nb; trial_Nb >= Nb_min; trial_Nb = (trial_Nb > Nb_min * 2) ? std::max(Nb_min, trial_Nb / 2) : trial_Nb - vec_w) {
        int Nb_try = align_vec_up(std::min(trial_Nb, colsC));
        if (Nb_try > colsC) Nb_try = align_vec_down(colsC);
        int nb_blocks = (colsC + Nb_try - 1) / Nb_try;

        int best_Rb_for_Nb = 0;
        int best_Kc_for_Nb = Kc_min;

        int Kc_try = std::min(Kc_max_init, colsA);
        while (Kc_try >= Kc_min) {
            int Rb_try = estimate_Rb_for(Kc_try, Nb_try);
            if (Rb_try > best_Rb_for_Nb) {
                best_Rb_for_Nb = Rb_try;
                best_Kc_for_Nb = Kc_try;
            }
            if (Kc_try <= Kc_min) break;
            int next = Kc_try / 2;
            if (next < Kc_min && Kc_try - 1 >= Kc_min) next = Kc_try - 1;
            Kc_try = std::max(next, Kc_min);
        }

        if (best_Rb_for_Nb >= 1 && nb_blocks >= num_threads) {
            chosen_Nb = Nb_try;
            chosen_Kc = best_Kc_for_Nb;
            chosen_Rb = std::min(best_Rb_for_Nb, rowsA);
            found = true;
            break;
        }
    }

    if (!found) {
        int minimal_Nb_needed = (colsC + num_threads - 1) / num_threads;
        minimal_Nb_needed = align_vec_up(minimal_Nb_needed);
        if (minimal_Nb_needed < Nb_min) minimal_Nb_needed = Nb_min;
        if (minimal_Nb_needed > colsC) minimal_Nb_needed = align_vec_down(colsC);

        chosen_Nb = minimal_Nb_needed;
        int best_Rb_for_Nb = 0;
        int best_Kc_for_Nb = Kc_min;
        int Kc_try = std::min(Kc_max_init, colsA);
        while (Kc_try >= Kc_min) {
            int Rb_try = estimate_Rb_for(Kc_try, chosen_Nb);
            if (Rb_try > best_Rb_for_Nb) {
                best_Rb_for_Nb = Rb_try;
                best_Kc_for_Nb = Kc_try;
            }
            if (Kc_try <= Kc_min) break;
            int next = Kc_try / 2;
            if (next < Kc_min && Kc_try - 1 >= Kc_min) next = Kc_try - 1;
            Kc_try = std::max(next, Kc_min);
        }
        chosen_Kc = best_Kc_for_Nb;
        chosen_Rb = std::max(1, std::min(best_Rb_for_Nb, rowsA));
    }

    // final clamps
    chosen_Kc = std::max(Kc_min, std::min(chosen_Kc, colsA));
    chosen_Nb = std::max(Nb_min, std::min(chosen_Nb, colsC));
    chosen_Nb = align_vec_down(chosen_Nb);
    chosen_Rb = std::max(1, std::min(chosen_Rb, rowsA));

    Tiling t;
    t.Kc = chosen_Kc;
    t.Nb = chosen_Nb;
    t.Rb = chosen_Rb;
    t.num_threads = num_threads;
    return t;
}