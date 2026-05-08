/*
Kernels for the AdamW optimizer.

References:
  * https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  * https://github.com/nvidia/apex/blob/master/csrc/multi_tensor_adam.cu

Compile example:
nvcc adamw.cu -o adamw
nvcc -O3 --use_fast_math adamw.cu -o adamw

./adamw

TODO(general):
amsgrad=True

TODO(perf):
dtype
thread coarsening/ILP
*/

/*
计算的公式基本上就是下面的CPU版本的逻辑，默认就是求最小值，然后是修正
*/
void adamw_cpu(float *params, long long n, int t, float lr, float beta1, float beta2, float epsilon,
               float weight_decay, const float *grad, float *m, float *v, float *v_max, bool amsgrad) {
    // m, v默认都是0开始，所以才需要修正偏差
    // m_t = (1-β₁)·g_t + β₁·(1-β₁)·g_{t-1} + β₁²·(1-β₁)·g_{t-2} + ...
    // E[m_t] = E[g] · (1-β₁) · (1 + β₁ + β₁² + ... + β₁^{t-1})
    // = E[g] · (1 - β₁ᵗ)          ← 等比级数求和
    // 所以就是直接除以这个因子：
    // m̂_t = m_t / (1 - β₁ᵗ)
    // v̂_t = v_t / (1 - β₂ᵗ)
    float bias1 = 1.0f - powf(beta1, t);
    float bias2 = 1.0f - powf(beta2, t);
    for (int i = 0; i < n; ++i) {
        float n_paramter = params[i] - lr * weight_decay * params[i];
        float n_m = beta1 * m[i] + (1 - beta1) * grad[i];
        float n_v = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float n_m_hat = n_m / bias1;
        float n_v_hat;
        if (amsgrad) {
            /*
            * 由于旧的梯度信息会被指数级遗忘，如果在训练过程中出现了一个非常大且包含重要信息的梯度，但随后跟着许多非常小的信息量较少的梯度，$v_t$ 的值就会迅速减小
            * 分母变小，会导致有效学习率（步长）突然异常增大。这打破了随机优化算法中“学习率应该逐渐减小或有上界”的收敛条件，可能导致模型在训练后期出现 Loss 剧烈震荡，甚至无法收敛到最优解。
            * AMSGrad 的思路非常直接且有效：强行记住出现过的最大二阶矩，防止分母变小。
            */
            float n_v_max = fmaxf(v_max[i], n_v);
            n_v_hat = n_v_max / bias2;
            v_max[i] = n_v_max;
        } else {
            n_v_hat = n_v / bias2;
        }
        n_paramter = n_paramter - lr * n_m_hat / (sqrtf(n_v_hat) + epsilon);
        m[i] = n_m;
        v[i] = n_v;
        params[i] = n_paramter;
    }
}

__global__ void adamw_gpu_fp32_1xn(float *params, long long n, float lr, float beta1, float beta2, float bias1,
                                   float bias2, float epsilon, float weight_decay, const float *grad, float *m, float *v,
                                   float *v_max, bool amsgrad) {
    // 一维网格
    if (const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; tid < n) {
        float n_parameter = params[tid] - lr * weight_decay * params[tid];
        const float n_m = beta1 * m[tid] + (1.0f - beta1) * grad[tid];
        const float n_v = beta2 * v[tid] + (1.0f - beta2) * grad[tid] * grad[tid];
        const float n_m_hat = n_m / bias1;
        float n_v_hat;
        if (amsgrad) {
            const float n_v_max = fmaxf(v_max[tid], n_v);
            n_v_hat = n_v_max / bias2;
            v_max[tid] = n_v_max;
        } else {
            n_v_hat = n_v / bias2;
        }
        n_parameter = n_parameter - lr * n_m_hat / (sqrtf(n_v_hat) + epsilon);
        m[tid] = n_m;
        v[tid] = n_v;
        params[tid] = n_parameter;
    }
}



