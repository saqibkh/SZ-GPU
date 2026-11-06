#ifndef SZGPU_HPP
#define SZGPU_HPP

#include "szgpu/szgpu.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace szgpu {

class Error : public std::runtime_error {
public:
    Error(szgpu_status_t status, const std::string& where)
        : std::runtime_error(where + ": " + szgpu_status_string(status)),
          status_(status) {}
    szgpu_status_t status() const noexcept { return status_; }
private:
    szgpu_status_t status_;
};

inline void check(szgpu_status_t s, const char* where) {
    if (s != SZGPU_STATUS_OK) throw Error(s, where);
}

class Workspace {
public:
    explicit Workspace(std::size_t max_n_elements) {
        check(szgpu_workspace_create(&ws_, max_n_elements), "szgpu_workspace_create");
    }
    ~Workspace() { if (ws_) szgpu_workspace_destroy(ws_); }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& other) noexcept : ws_(other.ws_) { other.ws_ = nullptr; }
    Workspace& operator=(Workspace&& other) noexcept {
        if (this != &other) {
            if (ws_) szgpu_workspace_destroy(ws_);
            ws_ = other.ws_; other.ws_ = nullptr;
        }
        return *this;
    }

    szgpu_workspace_t* handle() noexcept { return ws_; }
    std::size_t capacity() const noexcept { return szgpu_workspace_capacity(ws_); }

private:
    szgpu_workspace_t* ws_ = nullptr;
};

inline szgpu_compress_result_t compress(
    const float* d_in, unsigned char* d_cmp,
    std::size_t n, float abs_error_bound,
    Workspace& ws, cudaStream_t stream = 0)
{
    szgpu_compress_result_t r{};
    check(szgpu_compress(d_in, d_cmp, n, abs_error_bound, ws.handle(), stream, &r),
          "szgpu_compress");
    return r;
}

inline void decompress(
    float* d_out, const unsigned char* d_cmp,
    std::size_t n, float abs_error_bound,
    const szgpu_compress_result_t& result,
    Workspace& ws, cudaStream_t stream = 0)
{
    check(szgpu_decompress(d_out, d_cmp, n, abs_error_bound, &result, ws.handle(), stream),
          "szgpu_decompress");
}

inline szgpu_compress_result_t compress(
    const float* d_in, unsigned char* d_cmp,
    std::size_t n, float abs_error_bound, cudaStream_t stream = 0)
{
    szgpu_compress_result_t r{};
    check(szgpu_compress(d_in, d_cmp, n, abs_error_bound, nullptr, stream, &r),
          "szgpu_compress");
    return r;
}

inline szgpu_compress_result_t compress(
    const double* d_in, unsigned char* d_cmp,
    std::size_t n, double abs_error_bound, cudaStream_t stream = 0)
{
    szgpu_compress_result_t r{};
    check(szgpu_compress_f64(d_in, d_cmp, n, abs_error_bound, nullptr, stream, &r),
          "szgpu_compress_f64");
    return r;
}

inline void decompress(
    float* d_out, const unsigned char* d_cmp,
    std::size_t n, float abs_error_bound,
    const szgpu_compress_result_t& result, cudaStream_t stream = 0)
{
    check(szgpu_decompress(d_out, d_cmp, n, abs_error_bound, &result, nullptr, stream),
          "szgpu_decompress");
}

inline void decompress(
    double* d_out, const unsigned char* d_cmp,
    std::size_t n, double abs_error_bound,
    const szgpu_compress_result_t& result, cudaStream_t stream = 0)
{
    check(szgpu_decompress_f64(d_out, d_cmp, n, abs_error_bound, &result, nullptr, stream),
          "szgpu_decompress_f64");
}

inline std::size_t compress_hostptr(
    const float* in, unsigned char* cmp,
    std::size_t n, float abs_error_bound)
{
    std::size_t cmp_size = 0;
    check(szgpu_compress_hostptr(in, cmp, n, abs_error_bound, &cmp_size),
          "szgpu_compress_hostptr");
    return cmp_size;
}

inline void decompress_hostptr(
    float* out, const unsigned char* cmp, std::size_t cmp_size,
    std::size_t n, float abs_error_bound)
{
    check(szgpu_decompress_hostptr(out, cmp, cmp_size, n, abs_error_bound),
          "szgpu_decompress_hostptr");
}

inline szgpu_compress_result_t make_result(std::size_t n, std::size_t cmp_size) {
    return szgpu_make_result(n, cmp_size);
}

inline szgpu_compress_result_t compress(
    const double* d_in, unsigned char* d_cmp,
    std::size_t n, double abs_error_bound,
    Workspace& ws, cudaStream_t stream = 0)
{
    szgpu_compress_result_t r{};
    check(szgpu_compress_f64(d_in, d_cmp, n, abs_error_bound, ws.handle(), stream, &r),
          "szgpu_compress_f64");
    return r;
}

inline void decompress(
    double* d_out, const unsigned char* d_cmp,
    std::size_t n, double abs_error_bound,
    const szgpu_compress_result_t& result,
    Workspace& ws, cudaStream_t stream = 0)
{
    check(szgpu_decompress_f64(d_out, d_cmp, n, abs_error_bound, &result, ws.handle(), stream),
          "szgpu_decompress_f64");
}

inline std::size_t compress_hostptr(
    const double* in, unsigned char* cmp, std::size_t n, double abs_error_bound)
{
    std::size_t cmp_size = 0;
    check(szgpu_compress_hostptr_f64(in, cmp, n, abs_error_bound, &cmp_size),
          "szgpu_compress_hostptr_f64");
    return cmp_size;
}

inline void decompress_hostptr(
    double* out, const unsigned char* cmp, std::size_t cmp_size,
    std::size_t n, double abs_error_bound)
{
    check(szgpu_decompress_hostptr_f64(out, cmp, cmp_size, n, abs_error_bound),
          "szgpu_decompress_hostptr_f64");
}

inline std::size_t max_compressed_bytes(std::size_t n) { return szgpu_max_compressed_bytes(n); }
inline std::size_t num_tiles(std::size_t n)            { return szgpu_num_tiles(n); }
inline const char* version()                            { return szgpu_version(); }

}

#endif
