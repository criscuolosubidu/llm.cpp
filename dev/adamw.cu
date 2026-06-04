/*
Kernels for the AdamW optimizer.

References:
  * https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  * https://github.com/nvidia/apex/blob/master/csrc/multi_tensor_adam.cu

Compile example:
nvcc adamw.cu -o adamw
nvcc -O3 --use_fast_math adamw.cu -o adamw
*/
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include "utils.cuh"

using cuda_utils::DeviceBuffer;

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
        float n_parameter = params[i] - lr * weight_decay * params[i];
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
        n_parameter = n_parameter - lr * n_m_hat / (sqrtf(n_v_hat) + epsilon);
        m[i] = n_m;
        v[i] = n_v;
        params[i] = n_parameter;
    }
}

__global__ void adamw_gpu_fp32_1xn(float *params, long long n, float lr, float beta1, float beta2, float bias1,
                                   float bias2, float epsilon, float weight_decay, const float *grad, float *m,
                                   float *v,
                                   float *v_max, bool amsgrad) {
    // 一维网格
    if (const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; tid < n) {
        const float param = params[tid];
        const float o_m = m[tid];
        const float o_g = grad[tid];
        const float o_v = v[tid];
        float n_parameter = param - lr * weight_decay * param;
        const float n_m = beta1 * o_m + (1.0f - beta1) * o_g;
        const float n_v = beta2 * o_v + (1.0f - beta2) * o_g * o_g;
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

// https://developer.nvidia.com/blog/lerp-faster-cuda
// lerp: Linear interpolation，一种在两个已知点之间估算未知值的方法，假设两点之间的变化是均匀线性的, 也叫做线性插值
// lerp 使用非常广泛，所以一般有硬件的指令能够极其高效执行，但是一般使用的是低精度，
// 标准的线性插值的公式是：(1-t)*v0+t*v1
// 使用 fma (fused multiply-add，融和乘加）的操作计算，能够在一个周期内进行全精度的计算，那么可以转换为：
// fma(x, y, z) = x * y + z
// t*v1 - t*v0 + v0 = fma(t, v1, fma(-t, v0, v0))

__device__ __forceinline__ float lerp(float v0, float v1, float t) {
    return fma(t, v1, fma(-t, v0, v0));
}

// 使用lerp来稍微改进一下计算的速度
// m_i = beta1 * m_{i-1} + (1 - beta1) * g_t
// v_i = beta2 * v_{i-1} + (1 - beta2) * g_t^2

// use __restrict__ , 让编译器不再检查内存指针的重叠性问题
__global__ void adamw_gpu_fp32_1xn_lerp(
    float * __restrict__ params,
    long long n,
    float lr,
    float weight_decay_with_lr,
    float beta1,
    float beta2,
    float inv_bias1,
    float inv_bias2,
    float epsilon,
    const float * __restrict__ grad,
    float * __restrict__ m,
    float * __restrict__ v,
    float * __restrict__ v_max,
    bool amsgrad) {
    if (const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; tid < n) {
        const float param = params[tid];
        const float o_m = m[tid];
        const float o_g = grad[tid];
        const float o_v = v[tid];
        float n_parameter = param * weight_decay_with_lr;
        // p - lr * weight_decay * p = p * (1.0 - lr * weight_decay) = p * weight_decay_with_lr
        const float n_m = lerp(o_g, o_m, beta1);
        const float n_v = lerp(o_g * o_g, o_v, beta2);
        const float n_m_hat = n_m * inv_bias1;
        float n_v_hat;
        if (amsgrad) {
            const float n_v_max = fmaxf(v_max[tid], n_v);
            n_v_hat = n_v_max * inv_bias2;
            v_max[tid] = n_v_max;
        } else {
            n_v_hat = n_v * inv_bias2;
        }
        n_parameter = n_parameter - lr * n_m_hat / (sqrtf(n_v_hat) + epsilon);
        m[tid] = n_m;
        v[tid] = n_v;
        params[tid] = n_parameter;
    }
}

// 启动 adamw 优化器的配置
struct AdamWConfig {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
    bool amsgrad = false;
};

constexpr unsigned int ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

void launch_adamw(DeviceBuffer<float> &params, const DeviceBuffer<float> &grads, DeviceBuffer<float> &m,
                  DeviceBuffer<float> &v, DeviceBuffer<float> &v_max, int t, const AdamWConfig &config) {
    const uint64_t num_parameters = params.size();
    constexpr uint32_t block_size = 512;
    const uint32_t num_blocks = ceil_div(num_parameters, block_size);

    float weight_decay_with_lr = 1.0f - config.weight_decay * config.lr;
    float inv_bias1 = 1.0f / (1.0f - std::pow(config.beta1, t));
    float inv_bias2 = 1.0f / (1.0f - std::pow(config.beta2, t));

    adamw_gpu_fp32_1xn_lerp<<<num_blocks, block_size>>>(
        params.get(),
        num_parameters,
        config.lr,
        weight_decay_with_lr,
        config.beta1,
        config.beta2,
        inv_bias1,
        inv_bias2,
        config.eps,
        grads.get(),
        m.get(),
        v.get(),
        v_max.get(),
        config.amsgrad
    );

    CUDA_CHECK(cudaGetLastError());
}


std::vector<float> generate_random_vector(size_t size, float min_value = -1.0f, float max_value = 1.0f) {
    std::vector<float> vec(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_value, max_value);
    for (size_t i = 0; i < size; ++i) vec[i] = dis(gen);
    return vec;
}

int main() {
    try {
        constexpr size_t num_params = 1048'576;
        constexpr int t = 10;

        AdamWConfig config;
        config.amsgrad = true;

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";

        std::cout << "Global Memory: " << std::fixed << std::setprecision(1)
                  << (prop.totalGlobalMem / 1e9) << " GB\n";

        std::cout.unsetf(std::ios_base::floatfield);

        auto h_params = generate_random_vector(num_params);
        auto h_grads = generate_random_vector(num_params);
        auto h_m = generate_random_vector(num_params, 0.0f, 1.0f);
        auto h_v = generate_random_vector(num_params, 0.0f, 1.0f);
        auto h_v_max = std::vector<float>(num_params, 0.0f);

        DeviceBuffer<float> d_params(num_params);
        DeviceBuffer<float> d_grads(num_params);
        DeviceBuffer<float> d_m(num_params);
        DeviceBuffer<float> d_v(num_params);
        DeviceBuffer<float> d_v_max(num_params);

        d_params.copyFromHost(h_params);
        d_grads.copyFromHost(h_grads);
        d_m.copyFromHost(h_m);
        d_v.copyFromHost(h_v);
        if (config.amsgrad) d_v_max.zeroOut();

        launch_adamw(d_params, d_grads, d_m, d_v, d_v_max, t, config);
        CUDA_CHECK(cudaDeviceSynchronize());

        constexpr int repeat_times = 1000;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < repeat_times; ++i) {
            launch_adamw(d_params, d_grads, d_m, d_v, d_v_max, t, config);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        std::cout << "GPU kernel Executed " << repeat_times << " times.\n";
        std::cout << "Average time per launch: " << (elapsed.count() / repeat_times) << " ms\n";

        d_params.copyToHost(h_params);
        std::cout << "First updated params: " << h_params[0] << "\n";

    } catch (const std::exception &e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
