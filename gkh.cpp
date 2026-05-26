#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <arm_neon.h>
#include <pthread.h>
#include <string.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <cinttypes>   // for PRIx64
#include <omp.h>
#define NUM_THREADS 8

namespace
{

    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };
/*
    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            double a = M.at(r0, j);
            double b = M.at(r1, j);
            M.at(r0, j) = c * a + s * b;
            M.at(r1, j) = -s * a + c * b;
        }
    }
*/

    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        int cols = M.cols();
        double *row0 = &M.at(r0, 0);//行1的指针
        double *row1 = &M.at(r1, 0);

        float64x2_t vc = vdupq_n_f64(c);//c放进向量寄存器
        float64x2_t vs = vdupq_n_f64(s);
        float64x2_t vneg_s = vdupq_n_f64(-s);
        int j = 0;
        for (; j + 1 <cols; j+=2)//128位的寄存器，存2个double，一次处理2个
        {
            float64x2_t va = vld1q_f64(&row0[j]);//从内存加载2个double到向量寄存器
            float64x2_t vb = vld1q_f64(&row1[j]);

            float64x2_t new_row0 = vmlaq_f64(vmulq_f64(vc, va), vs, vb);//c*a + s*b，这是乘加指令，vmulq_f64(vc, va)计算c*a，vmlaq_f64在此基础上加上vs*vb，即s*b
            float64x2_t new_row1 = vmlaq_f64(vmulq_f64(vneg_s, va), vc, vb);

            vst1q_f64(&row0[j], new_row0);//写回内存
            vst1q_f64(&row1[j], new_row1);
        }
        for(; j < cols; ++j)//处理剩余的列
        {
            double a = M.at(r0, j);
            double b = M.at(r1, j);
            M.at(r0, j) = c * a + s * b;
            M.at(r1, j) = -s * a + c * b;
        }
    }


    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        for (int i = 0; i < M.rows(); ++i)
        {
            double a = M.at(i, c0);
            double b = M.at(i, c1);
            M.at(i, c0) = a * c - b * s;
            M.at(i, c1) = a * s + b * c;
        }
    }

    
    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <- U * L^T。
        // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // 由于 L^T = [c -s; s c]，此处复用“右乘两列”接口并传入 -s。
        apply_right_cols(U, r0, r1, c, -s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    struct Rot{
        int a;
        int b;
        double c;
        double s;
    };
    static void one_block_step(Matrix &B, int l, int r, std::vector<Rot> &u_log, std::vector<Rot> &v_log)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols(B, l, l + 1, c, s);
        //原apply_right_cols(V, l, l + 1, c, s);
        v_log.push_back({l, l + 1, c, -s});//记录V的旋转

        
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        u_log.push_back({l, l + 1, c, s});//记录U的旋转

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols(B, k, k + 1, c, s);
            //apply_right_cols(V, k, k + 1, c, s);
            v_log.push_back({k, k + 1, c, -s});      // 记录V的旋转
            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            //accumulate_left_into_U(U, k, k + 1, c, s);
            u_log.push_back({k, k + 1, c, s});       // 记录U的旋转
        }
    }


    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    // static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    // {
    //     if (r <= l)
    //     {
    //         return;
    //     }

    //     const double mu = block_wilkinson_shift(B, l, r);

    //     double c = 1.0;
    //     double s = 0.0;
    //     double rr = 0.0;

    //     // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
    //     const double x = B.at(l, l) * B.at(l, l) - mu;
    //     const double z = B.at(l, l) * B.at(l, l + 1);
    //     givens_rotation(x, z, c, s, rr, false);
    //     apply_right_cols(B, l, l + 1, c, s);
    //     //原apply_right_cols(V, l, l + 1, c, s);
    //     apply_left_rows(V, l, l + 1, c, -s); //新增
    //     // 首次左乘：消去 (l+1, l)。
    //     givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
    //     apply_left_rows(B, l, l + 1, c, s);
    //     //accumulate_left_into_U(U, l, l + 1, c, s);
    //     apply_left_rows(U, l, l + 1, c, s);  //新增
    //     for (int k = l + 1; k <= r - 1; ++k)
    //     {
    //         // 右乘：消去 (k-1, k+1)
    //         givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
    //         apply_right_cols(B, k, k + 1, c, s);
    //         //apply_right_cols(V, k, k + 1, c, s);
    //         apply_left_rows(V, k, k + 1, c, -s);      // V 行操作
    //         // 左乘：消去 (k+1, k)
    //         givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
    //         apply_left_rows(B, k, k + 1, c, s);
    //         //accumulate_left_into_U(U, k, k + 1, c, s);
    //         apply_left_rows(U, k, k + 1, c, s);       // U 行操作
    //     }
    // }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal(Matrix &B, int k, double tol, std::vector<Rot> &u_log, std::vector<Rot> &v_log)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols(B, i, i + 1, c, s);
            //apply_right_cols(V, i, i + 1, c, s);
            v_log.push_back({i, i + 1, c, -s});      // 记录V的旋转
            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                //accumulate_left_into_U(U, i, i + 1, c, s);
                u_log.push_back({i, i + 1, c, s});       // 记录U的旋转
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }
    // static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    // {
    //     const int m = B.rows();
    //     const int n = B.cols();
    //     if (k < 0 || k >= n - 1)
    //     {
    //         return false;
    //     }

    //     // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
    //     // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
    //     // 然后把问题传递到下一行，直到末端。
    //     if (std::fabs(B.at(k, k + 1)) <= tol)
    //     {
    //         return false;
    //     }

    //     bool changed = false;
    //     for (int i = k; i <= n - 2; ++i)
    //     {
    //         double c = 1.0;
    //         double s = 0.0;
    //         double rr = 0.0;

    //         // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
    //         givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
    //         apply_right_cols(B, i, i + 1, c, s);
    //         //apply_right_cols(V, i, i + 1, c, s);
    //         apply_left_rows(V, i, i + 1, c, -s);
    //         // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
    //         if (i + 1 < m)
    //         {
    //             givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
    //             apply_left_rows(B, i, i + 1, c, s);
    //             //accumulate_left_into_U(U, i, i + 1, c, s);
    //             apply_left_rows(U, i, i + 1, c, s);
    //         }

    //         changed = true;
    //     }

    //     cleanup_bidiagonal(B, tol);
    //     return changed;
    // }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros(Matrix &B, double tol,
                                  std::vector<Rot> &u_log, std::vector<Rot> &v_log) {
        const int n = B.cols();
        bool changed = false;
        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);
        for (int k = 0; k < n - 1; ++k) {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol) {
                if (chase_zero_diagonal(B, k, tol, u_log, v_log))
                    changed = true;
            }
        }
        return changed;
    }
    // static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    // {
    //     const int n = B.cols();
    //     bool changed = false;

    //     const double eps = std::numeric_limits<double>::epsilon();
    //     const double diag_tol = tol;
    //     const double super_tol = tol * (1.0 + 10.0 * eps);

    //     for (int k = 0; k < n - 1; ++k)
    //     {
    //         if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
    //         {
    //             if (chase_zero_diagonal(U, B, V, k, tol))
    //             {
    //                 changed = true;
    //             }
    //         }
    //     }

    //     return changed;
    // }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }
    
} // namespace

// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代（改进版）：
// - 输入输出满足 A = U * B * V^T 不变；
// - 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// - 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）。
/*----------------------------------------原始----------------------------------------------------------------------------*/
// bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
// {
//     const int m = B.rows();
//     const int n = B.cols();

//     if (m < n)
//     {
//         throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
//     }
//     if (U.rows() != m || U.cols() != m)
//     {
//         throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
//     }
//     if (V.rows() != n || V.cols() != n)
//     {
//         throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
//     }
//     //新增
//     U = U.transpose();
//     V = V.transpose();
//     bool converged = false;

//     for (int iter = 0; iter < max_iter; ++iter)
//     {
//         // 清理数值噪声，并优先处理 d_k≈0 的特殊情形。
//         cleanup_bidiagonal(B, tol);
//         handle_diagonal_zeros(U, B, V, tol);

//         // 根据超对角线断点拆分活动块
//         // 这里子矩阵间是相互独立的，所以此处具有很大的并行潜力：你可以尝试多线程/多进程进行处理
//         // 但根据算法，收集 Givens 旋转并更新 U/V 需要在每个块内顺序执行，所以这可能给并行带来麻烦。
//         std::vector<Block> blocks = split_active_blocks(B, n, tol);

//         // 若全部是 1x1 块，说明所有超对角都已收敛为 0。
//         bool all_singletons = true;
//         for (const auto &blk : blocks)
//         {
//             if (blk.r > blk.l)
//             {
//                 all_singletons = false;
//                 break;
//             }
//         }

//         if (all_singletons)
//         {
//             converged = true;
//             break;
//         }
        
//         //---------------------------非静态-----------------------------
// //-------------------------------------original--------------------------

//         // 从右到左处理每个非平凡块，减少末端块对前面块的干扰。
//         for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
//         {
//             if (blocks[i].r > blocks[i].l)
//             {
//                 one_block_step(U, B, V, blocks[i].l, blocks[i].r);
//             }
//         }

//     }

//     // 迭代结束后统一结构清理与标准化输出。
//     cleanup_bidiagonal(B, tol);
//     for (int i = 0; i < n - 1; ++i)
//     {
//         B.at(i, i + 1) = 0.0;
//     }
//     U = U.transpose();
//     V = V.transpose();
//     make_nonnegative_and_sort(U, B, V);

//     return converged;
// }


/*----------------------pthread------------------------------------------------------*/
/*--------------------------------------------------------静态线程--------------------------------------------------------------------------*/
// struct Task { int l; int r; };

// // 统计结构体
// struct ThreadStat {
//     long long busy_ns = 0;      // 忙碌时间
//     long long wait_ns = 0;      // 等待时间
//     int task_count = 0;         // 处理的任务数
//     long long total_rows = 0;   // 处理的总行数
// };

// // 静态线程参数
// typedef struct {
//     int tid;
//     Matrix *U;
//     Matrix *B;
//     Matrix *V;
//     std::vector<Task> *tasks;
//     int start_idx;
//     int end_idx;
//     double tol;
//     ThreadStat *stat;
// } StaticThreadArg;

// // 静态线程工作函数
// void* static_worker(void* arg) {
//     StaticThreadArg* params = (StaticThreadArg*)arg;
//     auto birth = std::chrono::steady_clock::now();
//     long long busy_ns = 0;
//     int count = 0;
//     long long rows = 0;

//     for (int i = params->start_idx; i < params->end_idx; ++i) {
//         auto t1 = std::chrono::steady_clock::now();
//         one_block_step(*(params->U), *(params->B), *(params->V),
//                        params->tasks->at(i).l, params->tasks->at(i).r);
//         auto t2 = std::chrono::steady_clock::now();
//         busy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
//         count++;
//         rows += params->tasks->at(i).r - params->tasks->at(i).l + 1;
//     }

//     auto death = std::chrono::steady_clock::now();
//     auto alive_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(death - birth).count();

//     params->stat->busy_ns = busy_ns;
//     params->stat->wait_ns = alive_ns - busy_ns;
//     params->stat->task_count = count;
//     params->stat->total_rows = rows;
//     return nullptr;
// }

// // 主函数
// bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
// {
//     const int m = B.rows();
//     const int n = B.cols();

//     if (m < n)
//         throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
//     if (U.rows() != m || U.cols() != m)
//         throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
//     if (V.rows() != n || V.cols() != n)
//         throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");

//     U = U.transpose();
//     V = V.transpose();
//     bool converged = false;

//     for (int iter = 0; iter < max_iter; ++iter)
//     {
//         cleanup_bidiagonal(B, tol);
//         handle_diagonal_zeros(U, B, V, tol);

//         std::vector<Block> blocks = split_active_blocks(B, n, tol);
//         bool all_singletons = true;
//         for (const auto &blk : blocks)
//             if (blk.r > blk.l) { all_singletons = false; break; }
//         if (all_singletons) { converged = true; break; }

//         // 收集非平凡块为任务
//         std::vector<Task> tasks;
//         tasks.reserve(blocks.size());
//         for (const auto &blk : blocks)
//             if (blk.r > blk.l) tasks.push_back({blk.l, blk.r});

//         // 静态划分任务
//         int num_threads = NUM_THREADS;
//         int total_tasks = static_cast<int>(tasks.size());
//         int tasks_per_thread = total_tasks / num_threads;
//         int remainder = total_tasks % num_threads;

//         std::vector<pthread_t> threads(num_threads);
//         std::vector<StaticThreadArg> args(num_threads);
//         std::vector<ThreadStat> stats(num_threads);

//         int start = 0;
//         for (int i = 0; i < num_threads; ++i) {
//             int count = tasks_per_thread + (i < remainder ? 1 : 0);
//             args[i] = {i, &U, &B, &V, &tasks, start, start + count, tol, &stats[i]};
//             pthread_create(&threads[i], nullptr, static_worker, &args[i]);
//             start += count;
//         }

//         for (int i = 0; i < num_threads; ++i) {
//             pthread_join(threads[i], nullptr);
//         }

//         // 每轮输出各线程统计（输出到 stderr，可重定向至文件）
//         for (int i = 0; i < num_threads; ++i) {
//             std::fprintf(stderr,
//                 "[Iter %d Thread %d] tasks=%d rows=%lld busy=%.3f ms wait=%.3f ms\n",
//                 iter, i, stats[i].task_count, stats[i].total_rows,
//                 stats[i].busy_ns / 1e6, stats[i].wait_ns / 1e6);
//         }
//     }

//     cleanup_bidiagonal(B, tol);
//     for (int i = 0; i < n - 1; ++i) B.at(i, i + 1) = 0.0;
//     U = U.transpose();
//     V = V.transpose();
//     make_nonnegative_and_sort(U, B, V);

//     return converged;
// }


/*-----------------------------------------------动态线程池调度------------------------------------------*/

// struct Task { int l; int r; };  // 单个块

//     // 任务组：指向全局任务列表的一段连续区间
//     struct TaskGroup {
//         int start_idx;  // 在全局列表中的起始索引
//         int count;      // 包含的块数量
//     };

//     struct ThreadPool {
//         pthread_t* threads;
//         int num_threads;

//         TaskGroup* task_queue;
//         int queue_capacity;
//         std::atomic<int> head{0};
//         std::atomic<int> tail{0};

//         pthread_mutex_t queue_lock;
//         pthread_cond_t  queue_cond;

//         // 完成同步：剩余要完成的任务组数
//         std::atomic<int> remaining_groups{0};
//         pthread_mutex_t complete_lock;
//         pthread_cond_t  complete_cond;

//         std::atomic<int> shutdown{0};

//         // 指向全局任务列表和矩阵
//         std::vector<Task>* all_tasks;  // 主线程设置的全局列表
//         Matrix* U;
//         Matrix* B;
//         Matrix* V;
//         double tol;
//     };
// //=--------------------------用于测量负载均衡-------------------------------------
// void* worker(void* arg) {
//     ThreadPool* pool = static_cast<ThreadPool*>(arg);
    
//     // ---- 用于计时 ----
//     auto birth = std::chrono::steady_clock::now();
//     long long busy_ns = 0;

//     // ---- 用于负载统计 ----
//     int task_groups = 0;
//     int total_blocks = 0;
//     long long total_rows = 0;

//     while (true) {
//         pthread_mutex_lock(&pool->queue_lock);
//         while (pool->head.load(std::memory_order_relaxed) >=
//                    pool->tail.load(std::memory_order_relaxed) &&
//                !pool->shutdown.load(std::memory_order_relaxed)) {
//             pthread_cond_wait(&pool->queue_cond, &pool->queue_lock);
//         }
//         if (pool->shutdown.load(std::memory_order_relaxed)) {
//             pthread_mutex_unlock(&pool->queue_lock);
//             break;
//         }
//         TaskGroup group = pool->task_queue[pool->head.fetch_add(1, std::memory_order_relaxed)];
//         pthread_mutex_unlock(&pool->queue_lock);

//         auto task_start = std::chrono::steady_clock::now();

//         for (int i = 0; i < group.count; ++i) {
//             const Task &t = pool->all_tasks->at(group.start_idx + i);
//             int blk_sz = t.r - t.l + 1;
//             total_blocks++;
//             total_rows += blk_sz;
//             one_block_step(*(pool->U), *(pool->B), *(pool->V), t.l, t.r);
//         }

//         auto task_end = std::chrono::steady_clock::now();
//         busy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(task_end - task_start).count();
//         task_groups++;

//         if (pool->remaining_groups.fetch_sub(1, std::memory_order_release) == 1) {
//             pthread_mutex_lock(&pool->complete_lock);
//             pthread_cond_signal(&pool->complete_cond);
//             pthread_mutex_unlock(&pool->complete_lock);
//         }
//     }

//     auto death = std::chrono::steady_clock::now();
//     auto alive_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(death - birth).count();
//     long long wait_ns = alive_ns - busy_ns;

//     // 使用线程ID的哈希值作为标识
//     auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());

//     std::fprintf(stderr,
//         "[Thread %zu] alive=%10.3f s  busy=%10.3f s  wait=%10.3f s  "
//         "groups=%d  blocks=%d  rows=%lld\n",
//         tid,
//         alive_ns / 1e9,
//         busy_ns / 1e9,
//         wait_ns / 1e9,
//         task_groups,
//         total_blocks,
//         total_rows);

//     return nullptr;
// }
// //----------------------------实际代码

//     void* worker(void* arg) {
//         ThreadPool* pool = static_cast<ThreadPool*>(arg);
//         while (true) {

        
//             // 获取一个任务组
//             pthread_mutex_lock(&pool->queue_lock);
//             while (pool->head.load(std::memory_order_relaxed) >=
//                        pool->tail.load(std::memory_order_relaxed) &&
//                    !pool->shutdown.load(std::memory_order_relaxed)) {
//                 pthread_cond_wait(&pool->queue_cond, &pool->queue_lock);
//             }
//             if (pool->shutdown.load(std::memory_order_relaxed)) {
//                 pthread_mutex_unlock(&pool->queue_lock);
//                 break;
//             }
//             TaskGroup group = pool->task_queue[pool->head.fetch_add(1, std::memory_order_relaxed)];
//             pthread_mutex_unlock(&pool->queue_lock);

//             // 串行处理组内所有块
//             for (int i = 0; i < group.count; ++i) {
//                 const Task &t = pool->all_tasks->at(group.start_idx + i);
//                 one_block_step(*(pool->U), *(pool->B), *(pool->V), t.l, t.r);
//             }

//             // 完成一个组，递减剩余计数
//             if (pool->remaining_groups.fetch_sub(1, std::memory_order_release) == 1) {
//                 // 最后一个完成的组，唤醒主线程
//                 pthread_mutex_lock(&pool->complete_lock);
//                 pthread_cond_signal(&pool->complete_cond);
//                 pthread_mutex_unlock(&pool->complete_lock);
//             }
//         }
//         return nullptr;
//     }


//     void thread_pool_init(ThreadPool* pool, int num_threads,
//                           Matrix* U, Matrix* B, Matrix* V, double tol,
//                           int queue_capacity) {
//         pool->num_threads = num_threads;
//         pool->queue_capacity = queue_capacity;
//         pool->threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
//         pool->task_queue = (TaskGroup*)malloc(queue_capacity * sizeof(TaskGroup));
//         pool->head.store(0, std::memory_order_relaxed);
//         pool->tail.store(0, std::memory_order_relaxed);
//         pool->remaining_groups.store(0, std::memory_order_relaxed);
//         pool->shutdown.store(0, std::memory_order_relaxed);
//         pool->all_tasks = nullptr;
//         pool->U = U;
//         pool->B = B;
//         pool->V = V;
//         pool->tol = tol;

//         pthread_mutex_init(&pool->queue_lock, nullptr);
//         pthread_cond_init(&pool->queue_cond, nullptr);
//         pthread_mutex_init(&pool->complete_lock, nullptr);
//         pthread_cond_init(&pool->complete_cond, nullptr);

//         for (int i = 0; i < num_threads; ++i) {
//             if (pthread_create(&pool->threads[i], nullptr, worker, pool) != 0) {
//                 std::fprintf(stderr, "Error creating thread %d\n", i);
//                 std::exit(1);
//             }
//         }
//     }

//     // 将全局任务列表按固定组大小划分为多个 TaskGroup
//     void thread_pool_execute(ThreadPool* pool, std::vector<Task>& all_tasks) {
//         if (all_tasks.empty()) return;

//         // 分组策略：每组最多 TASKS_PER_GROUP 个块
//         const int TASKS_PER_GROUP = 8; 
//         std::vector<TaskGroup> groups;
//         groups.reserve((all_tasks.size() + TASKS_PER_GROUP - 1) / TASKS_PER_GROUP);
//         for (size_t i = 0; i < all_tasks.size(); i += TASKS_PER_GROUP) {
//             int count = std::min((int)(all_tasks.size() - i), TASKS_PER_GROUP);
//             groups.push_back({(int)i, count});
//         }

//         // 设置全局任务列表指针
//         pool->all_tasks = &all_tasks;

//         // 入队所有任务组
//         pthread_mutex_lock(&pool->queue_lock);
//         int avail = pool->queue_capacity - (pool->tail.load(std::memory_order_relaxed) - pool->head.load(std::memory_order_relaxed));
//         if ((int)groups.size() > avail) {
//             std::fprintf(stderr, "Task queue overflow. Increase queue_capacity.\n");
//             std::exit(1);
//         }
//         for (const auto& g : groups) {
//             pool->task_queue[pool->tail.fetch_add(1, std::memory_order_relaxed)] = g;
//         }
//         pool->remaining_groups.store((int)groups.size(), std::memory_order_release);
//         pthread_cond_broadcast(&pool->queue_cond);
//         pthread_mutex_unlock(&pool->queue_lock);

//         // 等待所有组完成
//         pthread_mutex_lock(&pool->complete_lock);
//         while (pool->remaining_groups.load(std::memory_order_acquire) > 0) {
//             pthread_cond_wait(&pool->complete_cond, &pool->complete_lock);
//         }
//         pthread_mutex_unlock(&pool->complete_lock);

//         // 重置队列（所有组已完成）
//         pthread_mutex_lock(&pool->queue_lock);
//         pool->head.store(0, std::memory_order_relaxed);
//         pool->tail.store(0, std::memory_order_relaxed);
//         pthread_mutex_unlock(&pool->queue_lock);
//     }

//     void thread_pool_destroy(ThreadPool* pool) {
//         pthread_mutex_lock(&pool->queue_lock);
//         pool->shutdown.store(1, std::memory_order_relaxed);
//         pthread_cond_broadcast(&pool->queue_cond);
//         pthread_mutex_unlock(&pool->queue_lock);

//         for (int i = 0; i < pool->num_threads; ++i)
//             pthread_join(pool->threads[i], nullptr);

//         free(pool->threads);
//         free(pool->task_queue);
//         pthread_mutex_destroy(&pool->queue_lock);
//         pthread_cond_destroy(&pool->queue_cond);
//         pthread_mutex_destroy(&pool->complete_lock);
//         pthread_cond_destroy(&pool->complete_cond);
//     }


    
//     
//     bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V,
//                                  int max_iter, double tol) {
//         const int m = B.rows(), n = B.cols();
//         if (m < n) throw std::invalid_argument("m >= n required");
//         if (U.rows() != m || U.cols() != m) throw std::invalid_argument("U size mismatch");
//         if (V.rows() != n || V.cols() != n) throw std::invalid_argument("V size mismatch");

//         // 转置
//         U = U.transpose();
//         V = V.transpose();

//         // 初始化线程池，队列容量设为最大可能的组数（n足够）
//         ThreadPool pool;
//         thread_pool_init(&pool, NUM_THREADS, &U, &B, &V, tol, std::max(1024, n * 2));

//         bool converged = false;
        


//         // 在函数开头声明累计时间变量
// long long total_serial_ns = 0;
// long long total_parallel_ns = 0;

// for (int iter = 0; iter < max_iter; ++iter) {
//     auto iter_start = std::chrono::steady_clock::now();

//     // 串行部分
//     cleanup_bidiagonal(B, tol);
//     handle_diagonal_zeros(U, B, V, tol);

//     auto after_serial = std::chrono::steady_clock::now();

//     // 分块
//     std::vector<Block> blocks = split_active_blocks(B, n, tol);

//     // 收敛检查
//     bool all_singletons = true;
//     for (const auto &blk : blocks)
//         if (blk.r > blk.l) { all_singletons = false; break; }
//     if (all_singletons) { converged = true; break; }

//     // 收集任务
//     std::vector<Task> tasks;
//     tasks.reserve(blocks.size());
//     for (const auto &blk : blocks)
//         if (blk.r > blk.l) tasks.push_back({blk.l, blk.r});

//     // 并行部分
//     auto before_parallel = std::chrono::steady_clock::now();
//     thread_pool_execute(&pool, tasks);
//     auto after_parallel = std::chrono::steady_clock::now();

//     // 统计本轮时间
//     long long serial_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
//         after_serial - iter_start).count();
//     long long parallel_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
//         after_parallel - before_parallel).count();
//     total_serial_ns += serial_ns;
//     total_parallel_ns += parallel_ns;

//     // 计算最大块
//     int max_blk = 0;
//     for (const auto &blk : blocks) {
//         if (blk.r > blk.l) {
//             int sz = blk.r - blk.l + 1;
//             if (sz > max_blk) max_blk = sz;
//         }
//     }

//     // 输出迭代日志到 stderr
//     std::fprintf(stderr,
//         "[Iter %d] blocks=%zu  max_block=%d  serial=%.3f ms  parallel=%.3f ms\n",
//         iter, blocks.size(), max_blk,
//         serial_ns / 1.0e6, parallel_ns / 1.0e6);
// }

// // 循环结束后输出总计
// std::fprintf(stderr,
//     "=== Total serial time: %.3f s  total parallel time: %.3f s ===\n",
//     total_serial_ns / 1.0e9, total_parallel_ns / 1.0e9);



        
//         thread_pool_destroy(&pool);

//         // 收尾
//         cleanup_bidiagonal(B, tol);
//         for (int i = 0; i < n - 1; ++i) B.at(i, i + 1) = 0.0;
//         U = U.transpose();
//         V = V.transpose();
//         make_nonnegative_and_sort(U, B, V);
//         return converged;
//     }

//----------------------openmp------------------------------------------------------
/*
//-------------------------实际算法-----------------------
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }
    //新增
    U = U.transpose();
    V = V.transpose();
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // 清理数值噪声，并优先处理 d_k≈0 的特殊情形。
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        // 根据超对角线断点拆分活动块
        // 这里子矩阵间是相互独立的，所以此处具有很大的并行潜力：你可以尝试多线程/多进程进行处理
        // 但根据算法，收集 Givens 旋转并更新 U/V 需要在每个块内顺序执行，所以这可能给并行带来麻烦。
        std::vector<Block> blocks = split_active_blocks(B, n, tol);

        // 若全部是 1x1 块，说明所有超对角都已收敛为 0。
        bool all_singletons = true;
        for (const auto &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                all_singletons = false;
                break;
            }
        }

        if (all_singletons)
        {
            converged = true;
            break;
        }
        
        //---------------------------非静态-----------------------------
//-------------------------------------original--------------------------
        // 收集活跃块，并按大小降序排序（大块优先）
        std::vector<Block> active_blocks;
        for (const auto &blk : blocks)
            if (blk.r > blk.l)
                active_blocks.push_back(blk);
        std::sort(active_blocks.begin(), active_blocks.end(),
                  [](const Block& a, const Block& b) {
                      return (a.r - a.l) > (b.r - b.l);
                  });
        // 从右到左处理每个非平凡块，减少末端块对前面块的干扰。
        //#pragma omp parallel for schedule(dynamic,1)
        int num_threads = omp_get_max_threads();
        int num_tasks = active_blocks.size();
        int desired_groups = num_threads * 2; // 每线程2个任务
        int tasks_per_group = std::max(1,num_tasks / desired_groups);
        if(tasks_per_group * desired_groups < num_tasks){
            tasks_per_group += 1;
        }
        struct Group{int start; int count;};
        std::vector<Group> groups;
        for(int i = 0; i < num_tasks; i += tasks_per_group){
            int count = std::min(tasks_per_group, num_tasks - i);
            groups.push_back({i, count});
        }
        #pragma omp parallel for schedule(dynamic,1)
        for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
        {
            if (blocks[i].r > blocks[i].l)
            {
                one_block_step(U, B, V, blocks[i].l, blocks[i].r);
            }
        }

    }

    // 迭代结束后统一结构清理与标准化输出。
    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }
    U = U.transpose();
    V = V.transpose();
    make_nonnegative_and_sort(U, B, V);

    return converged;
}
*/
/*
//----------------------------测试用-------------------------------------------------------
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }
    //新增
    U = U.transpose();
    V = V.transpose();
    bool converged = false;

    // 函数开头声明累计时间
long long total_serial_ns = 0;
long long total_parallel_ns = 0;

for (int iter = 0; iter < max_iter; ++iter) {
    auto iter_start = std::chrono::steady_clock::now();

    // 串行部分：清理与对角零追赶
    cleanup_bidiagonal(B, tol);
    handle_diagonal_zeros(U, B, V, tol);

    auto after_serial = std::chrono::steady_clock::now();

    // 分块
    std::vector<Block> blocks = split_active_blocks(B, n, tol);

    // 收敛检查
    bool all_singletons = true;
    for (const auto &blk : blocks)
        if (blk.r > blk.l) { all_singletons = false; break; }
    if (all_singletons) { converged = true; break; }


    std::vector<Block> active_blocks;
    for (const auto &blk : blocks)
        if (blk.r > blk.l) active_blocks.push_back(blk);
    std::sort(active_blocks.begin(), active_blocks.end(),
              [](const Block& a, const Block& b) {
                  return (a.r - a.l) > (b.r - b.l);
              });

    // 动态分组
    int num_threads = omp_get_max_threads();
    int num_tasks = active_blocks.size();
    int desired_groups = num_threads * 2;
    int tasks_per_group = std::max(1, num_tasks / desired_groups);
    if (tasks_per_group * desired_groups < num_tasks) tasks_per_group++;

    struct Group { int start; int count; };
    std::vector<Group> groups;
    for (int i = 0; i < num_tasks; i += tasks_per_group) {
        int cnt = std::min(tasks_per_group, num_tasks - i);
        groups.push_back({i, cnt});
    }

    // 并行部分
    auto before_parallel = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        const Group &grp = groups[g];
        for (int k = 0; k < grp.count; ++k) {
            const Block &blk = active_blocks[grp.start + k];
            one_block_step(U, B, V, blk.l, blk.r);
        }
    }

    auto after_parallel = std::chrono::steady_clock::now();

    // 统计本轮时间
    long long serial_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        after_serial - iter_start).count();
    long long parallel_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        after_parallel - before_parallel).count();
    total_serial_ns += serial_ns;
    total_parallel_ns += parallel_ns;

    // 计算最大块
    int max_blk = 0;
    for (const auto &blk : blocks) {
        if (blk.r > blk.l) {
            int sz = blk.r - blk.l + 1;
            if (sz > max_blk) max_blk = sz;
        }
    }

    // 输出迭代日志到 stderr
    std::fprintf(stderr,
        "[Iter %d] blocks=%zu  max_block=%d  serial=%.3f ms  parallel=%.3f ms\n",
        iter, blocks.size(), max_blk,
        serial_ns / 1.0e6, parallel_ns / 1.0e6);
}

// 循环结束后输出总计
std::fprintf(stderr,
    "=== Total serial time: %.3f s  total parallel time: %.3f s ===\n",
    total_serial_ns / 1.0e9, total_parallel_ns / 1.0e9);

    // 迭代结束后统一结构清理与标准化输出。
    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }
    U = U.transpose();
    V = V.transpose();
    make_nonnegative_and_sort(U, B, V);

    return converged;
}
*/
    

    // ---------- 线程池（仅操作 B，记录旋转）----------
    struct Task {
        int l, r;
        std::vector<Rot> u_rots;
        std::vector<Rot> v_rots;
    };

    struct TaskGroup { int start_idx; int count; };

    struct ThreadPool {
        pthread_t* threads;
        int num_threads;
        TaskGroup* task_queue;
        int queue_capacity;
        std::atomic<int> head{0}, tail{0};
        pthread_mutex_t queue_lock;
        pthread_cond_t  queue_cond;
        std::atomic<int> remaining_groups{0};
        pthread_mutex_t complete_lock;
        pthread_cond_t  complete_cond;
        std::atomic<int> shutdown{0};
        std::vector<Task>* all_tasks;
        Matrix* B;          // 工作线程只操作 B
        double tol;
    };

    void* worker(void* arg) {
        ThreadPool* pool = static_cast<ThreadPool*>(arg);
        while (true) {
            pthread_mutex_lock(&pool->queue_lock);
            while (pool->head.load(std::memory_order_relaxed) >=
                       pool->tail.load(std::memory_order_relaxed) &&
                   !pool->shutdown.load(std::memory_order_relaxed)) {
                pthread_cond_wait(&pool->queue_cond, &pool->queue_lock);
            }
            if (pool->shutdown.load(std::memory_order_relaxed)) {
                pthread_mutex_unlock(&pool->queue_lock);
                break;
            }
            TaskGroup group = pool->task_queue[pool->head.fetch_add(1, std::memory_order_relaxed)];
            pthread_mutex_unlock(&pool->queue_lock);

            for (int i = 0; i < group.count; ++i) {
                Task &t = (*pool->all_tasks)[group.start_idx + i];
                one_block_step(*(pool->B), t.l, t.r, t.u_rots, t.v_rots);
            }

            if (pool->remaining_groups.fetch_sub(1, std::memory_order_release) == 1) {
                pthread_mutex_lock(&pool->complete_lock);
                pthread_cond_signal(&pool->complete_cond);
                pthread_mutex_unlock(&pool->complete_lock);
            }
        }
        return nullptr;
    }

    void thread_pool_init(ThreadPool* pool, int num_threads,
                          Matrix* B, double tol, int queue_capacity) {
        pool->num_threads = num_threads;
        pool->queue_capacity = queue_capacity;
        pool->threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        pool->task_queue = (TaskGroup*)malloc(queue_capacity * sizeof(TaskGroup));
        pool->head.store(0, std::memory_order_relaxed);
        pool->tail.store(0, std::memory_order_relaxed);
        pool->remaining_groups.store(0, std::memory_order_relaxed);
        pool->shutdown.store(0, std::memory_order_relaxed);
        pool->all_tasks = nullptr;
        pool->B = B;
        pool->tol = tol;

        pthread_mutex_init(&pool->queue_lock, nullptr);
        pthread_cond_init(&pool->queue_cond, nullptr);
        pthread_mutex_init(&pool->complete_lock, nullptr);
        pthread_cond_init(&pool->complete_cond, nullptr);

        for (int i = 0; i < num_threads; ++i) {
            if (pthread_create(&pool->threads[i], nullptr, worker, pool) != 0) {
                std::fprintf(stderr, "Error creating thread %d\n", i);
                std::exit(1);
            }
        }
    }

    void thread_pool_execute(ThreadPool* pool, std::vector<Task>& all_tasks) {
        if (all_tasks.empty()) return;
        const int TASKS_PER_GROUP = 8;
        std::vector<TaskGroup> groups;
        for (size_t i = 0; i < all_tasks.size(); i += TASKS_PER_GROUP) {
            int count = std::min((int)(all_tasks.size() - i), TASKS_PER_GROUP);
            groups.push_back({(int)i, count});
        }
        pool->all_tasks = &all_tasks;

        pthread_mutex_lock(&pool->queue_lock);
        for (const auto& g : groups)
            pool->task_queue[pool->tail.fetch_add(1, std::memory_order_relaxed)] = g;
        pool->remaining_groups.store((int)groups.size(), std::memory_order_release);
        pthread_cond_broadcast(&pool->queue_cond);
        pthread_mutex_unlock(&pool->queue_lock);

        pthread_mutex_lock(&pool->complete_lock);
        while (pool->remaining_groups.load(std::memory_order_acquire) > 0)
            pthread_cond_wait(&pool->complete_cond, &pool->complete_lock);
        pthread_mutex_unlock(&pool->complete_lock);

        pthread_mutex_lock(&pool->queue_lock);
        pool->head.store(0, std::memory_order_relaxed);
        pool->tail.store(0, std::memory_order_relaxed);
        pthread_mutex_unlock(&pool->queue_lock);
    }

    void thread_pool_destroy(ThreadPool* pool) {
        pthread_mutex_lock(&pool->queue_lock);
        pool->shutdown.store(1, std::memory_order_relaxed);
        pthread_cond_broadcast(&pool->queue_cond);
        pthread_mutex_unlock(&pool->queue_lock);
        for (int i = 0; i < pool->num_threads; ++i)
            pthread_join(pool->threads[i], nullptr);
        free(pool->threads);
        free(pool->task_queue);
        pthread_mutex_destroy(&pool->queue_lock);
        pthread_cond_destroy(&pool->queue_cond);
        pthread_mutex_destroy(&pool->complete_lock);
        pthread_cond_destroy(&pool->complete_cond);
    }

    // ========== 并行 GKH SVD 入口 ==========
    bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V,
                                 int max_iter, double tol) {
        const int m = B.rows(), n = B.cols();
        if (m < n) throw std::invalid_argument("m >= n required");
        if (U.rows() != m || U.cols() != m) throw std::invalid_argument("U size mismatch");
        if (V.rows() != n || V.cols() != n) throw std::invalid_argument("V size mismatch");

        int num_threads = NUM_THREADS;
        const char* env = std::getenv("SVD_THREADS");
        if (env) num_threads = std::atoi(env);
        if (num_threads < 1) num_threads = 1;

        U = U.transpose();
        V = V.transpose();

        ThreadPool pool;
        thread_pool_init(&pool, num_threads, &B, tol, std::max(1024, n * 2));

        bool converged = false;
        for (int iter = 0; iter < max_iter; ++iter) {
            // 对角零追赶（记录旋转）
            std::vector<Rot> diag_u, diag_v;
            handle_diagonal_zeros(B, tol, diag_u, diag_v);

            // 分块
            std::vector<Block> blocks = split_active_blocks(B, n, tol);
            bool all_singletons = true;
            for (const auto &blk : blocks)
                if (blk.r > blk.l) { all_singletons = false; break; }
            if (all_singletons) {
                for (const auto &r : diag_u) apply_left_rows(U, r.a, r.b, r.c, r.s);
                for (const auto &r : diag_v) apply_left_rows(V, r.a, r.b, r.c, r.s);
                converged = true;
                break;
            }

            // 准备任务（带空日志）
            std::vector<Task> tasks;
            tasks.reserve(blocks.size());
            for (const auto &blk : blocks)
                if (blk.r > blk.l) tasks.push_back({blk.l, blk.r, {}, {}});

            // 并行追赶（只操作 B，填充日志）
            thread_pool_execute(&pool, tasks);

            // 合并所有旋转日志
            std::vector<Rot> all_u = std::move(diag_u), all_v = std::move(diag_v);
            for (auto &task : tasks) {
                all_u.insert(all_u.end(), task.u_rots.begin(), task.u_rots.end());
                all_v.insert(all_v.end(), task.v_rots.begin(), task.v_rots.end());
            }

            // 批量并行应用旋转到 U/V
            for (const auto &r : all_u) apply_left_rows(U, r.a, r.b, r.c, r.s);
            for (const auto &r : all_v) apply_left_rows(V, r.a, r.b, r.c, r.s);
        }

        thread_pool_destroy(&pool);

        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i) B.at(i, i + 1) = 0.0;
        U = U.transpose();
        V = V.transpose();
        make_nonnegative_and_sort(U, B, V);

        return converged;
    }

    
