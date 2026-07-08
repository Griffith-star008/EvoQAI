#pragma once
// ============================================================================
// QHPC_memory_v10.hpp
// ----------------------------------------------------------------------------
// Minimal CUDA RAII helpers used throughout the QHPC v10+ codebase.
//
// Provides:
//   - CUDA_CHECK(expr): runtime CUDA error guard
//   - DeviceBuffer<T>:  RAII typed device-memory wrapper with:
//        .get()     -> raw T* device pointer
//        .resize(n) -> (re)allocate n elements
//        .zero(s)   -> async memset to zero on stream s
//        .upload(host_ptr, n, s)  -> async H2D copy
//        .emplace_back(args...)    -> append (for vector-of-buffers idiom in optimizer)
//        operator T*() implicit cast to raw pointer for kernel arg lists
//
// Self-contained — only needs <cuda_runtime.h>.
// ============================================================================
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _qhpc_cuda_err = (expr);                                  \
        if (_qhpc_cuda_err != cudaSuccess) {                                  \
            fprintf(stderr, "[CUDA] %s at %s:%d -> %s\n",                     \
                    #expr, __FILE__, __LINE__,                                \
                    cudaGetErrorString(_qhpc_cuda_err));                      \
            std::abort();                                                     \
        }                                                                     \
    } while (0)
#endif

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t n) { resize(n); }

    // No copy. Move-only.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            _free();
            ptr_ = other.ptr_;
            n_   = other.n_;
            other.ptr_ = nullptr;
            other.n_   = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { _free(); }

    void resize(size_t n) {
        if (n == n_ && ptr_) return;
        _free();
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
            n_ = n;
        }
    }

    void zero(cudaStream_t stream = nullptr) {
        if (ptr_ && n_ > 0) {
            CUDA_CHECK(cudaMemsetAsync(ptr_, 0, n_ * sizeof(T), stream));
        }
    }

    void upload(const T* host_src, size_t n, cudaStream_t stream = nullptr) {
        if (n == 0) return;
        if (n > n_) resize(n);
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host_src, n * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
    }

    void download(T* host_dst, size_t n, cudaStream_t stream = nullptr) const {
        if (n == 0 || !ptr_) return;
        CUDA_CHECK(cudaMemcpyAsync(host_dst, ptr_, n * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream));
    }

    T*       get()       { return ptr_; }
    const T* get() const { return ptr_; }
    size_t   size() const { return n_; }
    bool     empty() const { return n_ == 0 || ptr_ == nullptr; }

    // Implicit cast so a DeviceBuffer<T> can be passed where T* is expected.
    operator T*()             { return ptr_; }
    operator const T*() const { return ptr_; }

private:
    T*     ptr_ = nullptr;
    size_t n_   = 0;

    void _free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        n_ = 0;
    }
};

// Compatibility shim for the LBFGS optimizer which calls .emplace_back on
// a std::vector<DeviceBuffer<float>>. std::vector already supports emplace_back
// via DeviceBuffer's constructor — this is provided as a free helper for
// any other code path that expected a member-style call.
template <typename T, typename... Args>
inline void qhpc_emplace_buffer(std::vector<DeviceBuffer<T>>& vec, Args&&... args) {
    vec.emplace_back(std::forward<Args>(args)...);
}
