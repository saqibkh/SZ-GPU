#ifndef SZGPU_H
#define SZGPU_H

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SZGPU_VERSION_MAJOR 1
#define SZGPU_VERSION_MINOR 0
#define SZGPU_VERSION_PATCH 1
#define SZGPU_VERSION_STRING "1.0.1"

typedef enum {
    SZGPU_STATUS_OK = 0,
    SZGPU_STATUS_INVALID_ARGUMENT = 1,
    SZGPU_STATUS_OUT_OF_MEMORY = 2,
    SZGPU_STATUS_WORKSPACE_TOO_SMALL = 3,
    SZGPU_STATUS_CUDA_ERROR = 4,
    SZGPU_STATUS_TOO_LARGE = 5
} szgpu_status_t;

typedef struct szgpu_workspace szgpu_workspace_t;

typedef struct {
    size_t cmp_size;
    size_t num_tiles;
    size_t data_offset;
} szgpu_compress_result_t;

const char* szgpu_version(void);
const char* szgpu_status_string(szgpu_status_t status);

size_t szgpu_num_tiles(size_t n_elements);
size_t szgpu_max_compressed_bytes(size_t n_elements);

szgpu_status_t szgpu_workspace_create(szgpu_workspace_t** out_ws, size_t max_n_elements);
void szgpu_workspace_destroy(szgpu_workspace_t* ws);
size_t szgpu_workspace_capacity(const szgpu_workspace_t* ws);

szgpu_status_t szgpu_compress(
    const float* d_in,
    unsigned char* d_cmp,
    size_t n_elements,
    float abs_error_bound,
    szgpu_workspace_t* ws,
    cudaStream_t stream,
    szgpu_compress_result_t* out_result);

szgpu_status_t szgpu_decompress(
    float* d_out,
    const unsigned char* d_cmp,
    size_t n_elements,
    float abs_error_bound,
    const szgpu_compress_result_t* result,
    szgpu_workspace_t* ws,
    cudaStream_t stream);

szgpu_compress_result_t szgpu_make_result(size_t n_elements, size_t cmp_size);

szgpu_status_t szgpu_compress_hostptr(
    const float* in,
    unsigned char* cmp,
    size_t n_elements,
    float abs_error_bound,
    size_t* out_cmp_size);

szgpu_status_t szgpu_decompress_hostptr(
    float* out,
    const unsigned char* cmp,
    size_t cmp_size,
    size_t n_elements,
    float abs_error_bound);

szgpu_status_t szgpu_compress_f64(
    const double* d_in,
    unsigned char* d_cmp,
    size_t n_elements,
    double abs_error_bound,
    szgpu_workspace_t* ws,
    cudaStream_t stream,
    szgpu_compress_result_t* out_result);

szgpu_status_t szgpu_decompress_f64(
    double* d_out,
    const unsigned char* d_cmp,
    size_t n_elements,
    double abs_error_bound,
    const szgpu_compress_result_t* result,
    szgpu_workspace_t* ws,
    cudaStream_t stream);

szgpu_status_t szgpu_compress_hostptr_f64(
    const double* in,
    unsigned char* cmp,
    size_t n_elements,
    double abs_error_bound,
    size_t* out_cmp_size);

szgpu_status_t szgpu_decompress_hostptr_f64(
    double* out,
    const unsigned char* cmp,
    size_t cmp_size,
    size_t n_elements,
    double abs_error_bound);

#ifdef __cplusplus
}
#endif

#endif
