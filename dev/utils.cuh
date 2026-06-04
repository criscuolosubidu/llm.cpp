//
// Created by Administrator on 2026/6/3.
//

#ifndef LLM_CPP_UTILS_H
#define LLM_CPP_UTILS_H

#endif //LLM_CPP_UTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_utils {
    inline void checkCuda(cudaError_t result, const char* file, int line) {
        if (result != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Runtime Error: ") + cudaGetErrorString(result) + " at " + file + ":" + std::to_string(line));
        }
    }

#define CUDA_CHECK(val) cuda_utils::checkCuda(val, __FILE__, __LINE__)

    template<typename T>
    class DeviceBuffer {
    private:
        T* d_ptr = nullptr;
        size_t size_ = 0;
    public:
        explicit DeviceBuffer(size_t size): size_(size) {
            if (size > 0) {
                CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
            }
        }

        ~DeviceBuffer() {
            if (d_ptr) cudaFree(d_ptr);
        }

        // 禁止使用拷贝语义
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        // 允许移动语义
        DeviceBuffer(DeviceBuffer&& other) noexcept : d_ptr(other.d_ptr), size_(other.size_) {
            other.d_ptr = nullptr;
            other.size_ = 0;
        }

        void copyFromHost(const std::vector<T>& host_vec) {
            if (host_vec.size() != size_) throw std::invalid_argument("Size mismatch during copyToDevice");
            CUDA_CHECK(cudaMemcpy(d_ptr, host_vec.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
        }

        void copyToHost(std::vector<T>& host_vec) const {
            if (host_vec.size() != size_) throw std::invalid_argument("Size mismatch during copyToHost");
            CUDA_CHECK(cudaMemcpy(host_vec.data(), d_ptr, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void zeroOut() {
            CUDA_CHECK(cudaMemset(d_ptr, 0, size_ * sizeof(T)));
        }

        T* get() {
            return d_ptr;
        }
        const T* get() const {
            return d_ptr;
        }
        size_t size() const {
            return size_;
        }
    };
}

