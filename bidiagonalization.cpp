// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include<arm_neon.h>
// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double x : v)
        sum += x * x;
    return std::sqrt(sum);
}

// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            // sign(x[0])：此处规定 x[0]==0 时取 +1
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
            // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
            std::vector<double> v(x);
            v[0] += sigma; // v = x + sigma * e_1

            // 计算 v^T v
            double vTv = 0.0;
            for (double vi : v)
                vTv += vi * vi;
/*------------------------------------------------------------------------------------------*/
/*
            // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                std::vector<double> w(n - k, 0.0);
                for (int j = 0; j < n - k; ++j)
                    for (int i = 0; i < m - k; ++i)
                        w[j] += v[i] * B.at(k + i, k + j);
                for (int i = 0; i < m - k; ++i)
                    for (int j = 0; j < n - k; ++j)
                        B.at(k + i, k + j) -= beta * v[i] * w[j];

                // 累积 U：U_new = U_old * H_k
                // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                std::vector<double> wU(m, 0.0);
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m - k; ++j)
                        wU[i] += U.at(i, k + j) * v[j];
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m - k; ++j)
                        U.at(i, k + j) -= beta * wU[i] * v[j];
            }
*/


            //此处优化的思路是：
            //1.观察到B矩阵的访存不连续，于是我们调整为行主次序，使B命中率更高
            //2.进行SIMD向量化，两个64位double放一起作为向量进行运算
            if(vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                const int len = n - k;//w的长度
                const int rows = m - k;//B的行数
                //上面两个是为了简化表达
                std::vector<double> w(len, 0.0);
                //优化1：优化B的访问，变为行主次序
                for (int i = 0; i < rows; ++i){
                    double vi = v[i];
                    float64x2_t vvi = vdupq_n_f64(vi);//把v[i]放进向量寄存器
                    int j = 0;
                    for(; j+1 < len; j+=2){
                        float64x2_t b_row = vld1q_f64(&B.at(k + i, k + j));//加载B的第i行的第j列和第j+1列
                        float64x2_t w_val = vld1q_f64(&w[j]);//加载w的第j列和第j+1列
                        w_val = vmlaq_f64(w_val, vvi, b_row);//w[j] += v[i] * B.at(k + i, k + j) 和 w[j
                        vst1q_f64(&w[j], w_val);//写回w 
                    }
                    //处理奇数情况下剩余的部分
                    for(; j < len; ++j){
                        w[j] += vi * B.at(k + i, k + j);
                    }
                }
                //B-=beta*v*w^T
                for (int i = 0; i < rows; ++i){
                    double factor = beta * v[i];
                    float64x2_t v_factor = vdupq_n_f64(factor);
                    int j = 0;
                    for(; j+1 < len; j+=2){
                        float64x2_t b_row = vld1q_f64(&B.at(k + i, k + j));
                        float64x2_t w_val = vld1q_f64(&w[j]);
                        b_row = vmlsq_f64(b_row, v_factor, w_val);//B.at(k + i, k + j) -= factor * w[j] 和 B.at(k + i, k + j+1) -= factor * w[j+1]
                        vst1q_f64(&B.at(k + i, k + j), b_row);
                    }
                    for(; j < len; ++j){
                        B.at(k + i, k + j) -= factor * w[j];
                    }
                }
                // --- 累积 U : U(:, k:m) -= beta * (U(:, k:m) * v) * v^T ---
                // 注意：len 是 v 的长度 (m - k)
                const int lenU = rows;
                std::vector<double> wU(m, 0.0);
                for (int i = 0; i < m; ++i){
                    float64x2_t sum = vdupq_n_f64(0.0);
                    int j = 0;
                    for(; j+1 < lenU; j+=2){
                        float64x2_t u_vals = vld1q_f64(&U.at(i, k + j));
                        float64x2_t v_vals = vld1q_f64(&v[j]);
                        sum = vmlaq_f64(sum, u_vals, v_vals);//sum += U.at(i, k + j) * v[j] 和 sum += U.at(i, k +
                    }
                    double dot = vaddvq_f64(sum);//把sum的两个元素加起来得到点积的结果
                    for(; j < lenU; ++j){
                        dot += U.at(i, k + j) * v[j];
                    }
                    wU[i] = dot;
                }
                for(int i = 0;i < m; ++i){
                    double factor = beta * wU[i];
                    float64x2_t v_factor = vdupq_n_f64(factor);
                    int j = 0;
                    for(; j+1 < lenU; j+=2){
                        float64x2_t u_vals = vld1q_f64(&U.at(i, k + j));
                        float64x2_t v_vals = vld1q_f64(&v[j]);
                        u_vals = vmlsq_f64(u_vals, v_factor, v_vals);//U.at(i, k + j) -= factor * v[j] 和 U.at(i, k + j+1) -= factor * v[j+1]
                        vst1q_f64(&U.at(i, k + j), u_vals);
                    }
                    for(; j < lenU; ++j){
                        U.at(i, k + j) -= factor * v[j];
                    }
                }
            }


        }

/*------------------------------------------------------------------------------------------*/
        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = 0.0;
                for (double vi : v)
                    vTv += vi * vi;
//------------------------------------------------------------------------------------------*/
/*
                // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 注意：这里是从右侧作用 V_k
                    // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
                    std::vector<double> w(m - k, 0.0);
                    for (int i = 0; i < m - k; ++i)
                        for (int j = 0; j < n - k - 1; ++j)
                            w[i] += B.at(k + i, k + 1 + j) * v[j];
                    for (int i = 0; i < m - k; ++i)
                        for (int j = 0; j < n - k - 1; ++j)
                            B.at(k + i, k + 1 + j) -= beta * w[i] * v[j];

                    // 累积 V：V_new = V_old * V_k
                    // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    std::vector<double> wV(n, 0.0);
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < n - k - 1; ++j)
                            wV[i] += V.at(i, k + 1 + j) * v[j];
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < n - k - 1; ++j)
                            V.at(i, k + 1 + j) -= beta * wV[i] * v[j];
                }
*/


                //优化思路同上
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                    // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                    const int len = n - k - 1;//w的长度
                    const int rows = m - k;//B的行数
                    //上面两个是为了简化表达
                    std::vector<double> w(rows, 0.0);
                    for(int i = 0;i < rows;++i){
                        float64x2_t sum = vdupq_n_f64(0.0);
                        int j = 0;
                        for(; j+1 < len; j+=2){
                            float64x2_t b_vals = vld1q_f64(&B.at(k + i, k + 1 + j));
                            float64x2_t v_vals = vld1q_f64(&v[j]);
                            sum = vmlaq_f64(sum, b_vals, v_vals);//sum += B.at(k + i, k + 1 + j) * v[j] 和 sum += B.at(k + i, k + 1 + j+
                        }
                        double dot = vaddvq_f64(sum);
                        for(; j < len; ++j){
                            dot += B.at(k + i, k + 1 + j) * v[j];
                        }
                        w[i] = dot;

                    }
                    for(int i = 0;i < rows; ++i){
                        double factor = beta * w[i];
                        float64x2_t v_factor = vdupq_n_f64(factor);
                        int j = 0;
                        for(; j+1 < len; j+=2){
                            float64x2_t b_vals = vld1q_f64(&B.at(k + i, k + 1 + j));
                            float64x2_t v_vals = vld1q_f64(&v[j]);
                            b_vals = vmlsq_f64(b_vals, v_factor, v_vals);//B.at(k + i, k + 1 + j) -= factor * v[j] 和 B.at(k + i, k + 1 + j+1) -= factor * v[j+1]
                            vst1q_f64(&B.at(k + i, k + 1 + j), b_vals);
                        }
                        for(; j < len; ++j){
                            B.at(k + i, k + 1 + j) -= factor * v[j];
                        }
                    }
                    // --- 累积 V : V(:, k+1:n) -= beta * (V(:, k+1:n) * v) * v^T ---
                    // 注意：len 是 v 的长度 (n - k - 1)
                    std::vector<double> wV(n, 0.0);
                    for (int i = 0; i < n; ++i){
                        float64x2_t sum = vdupq_n_f64(0.0);
                        int j = 0;
                        for(; j+1 < len; j+=2){
                            float64x2_t V_vals = vld1q_f64(&V.at(i, k + 1 + j));
                            float64x2_t v_vals = vld1q_f64(&v[j]);
                            sum = vmlaq_f64(sum, V_vals, v_vals);//sum += V.at(i, k + 1 + j) * v[j] 和 sum += V.at(i, k + 1 + j+1) * v[j+1]
                        }
                        double dot = vaddvq_f64(sum);
                        for(; j < len; ++j){
                            dot += V.at(i, k + 1 + j) * v[j];
                        }
                        wV[i] = dot;
                    }
                    for(int i = 0;i<n;i++){
                        double factor = beta * wV[i];
                        float64x2_t v_factor = vdupq_n_f64(factor);
                        int j = 0;
                        for(; j+1 < len; j+=2){
                            float64x2_t V_vals = vld1q_f64(&V.at(i, k + 1 + j));
                            float64x2_t v_vals = vld1q_f64(&v[j]);
                            V_vals = vmlsq_f64(V_vals, v_factor, v_vals);//V.at(i, k + 1 + j) -= factor * v[j] 和 V.at(i, k + 1 + j+1) -= factor * v[j+1]
                            vst1q_f64(&V.at(i, k + 1 + j), V_vals);
                        }
                        for(; j < len; ++j){
                            V.at(i, k + 1 + j) -= factor * v[j];
                        }
                    }
                }

                

//------------------------------------------------------------------------------------------*/
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}
