#include "szgpu/szgpu.h"

#include "szgpu_internal.cuh"
#include "szgpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

using szgpu_detail::szgpu_compress_kernel_T;
using szgpu_detail::szgpu_decompress_kernel_T;

struct szgpu_workspace {
    unsigned int* d_workspace;
    unsigned int* cmpOffset;
    unsigned int* locOffset;
    int* flag;
    unsigned int* h_cmpSize;
    int gsize_max;
    int gsize_sm_max;
    int cmpOffSize;
    size_t max_n_elements;
};

static int szgpu_grid_size(size_t numTiles) {
    return (int)((numTiles + SZGPU_TBLOCK_SZ * SZGPU_TILES_PER_THR - 1)
                 / (SZGPU_TBLOCK_SZ * SZGPU_TILES_PER_THR));
}

static int szgpu_grid_size_sm(size_t numTiles) {
    return (int)((numTiles + SZGPU_TBLOCK_SZ * SZGPU_TILES_PER_THR_SM - 1)
                 / (SZGPU_TBLOCK_SZ * SZGPU_TILES_PER_THR_SM));
}

static void szgpu_hostptr_release(void* d_a, void* d_b, szgpu_workspace_t* ws) {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (ws)  szgpu_workspace_destroy(ws);
}

extern "C" const char* szgpu_version(void) {
    return SZGPU_VERSION_STRING;
}

extern "C" const char* szgpu_status_string(szgpu_status_t s) {
    switch (s) {
        case SZGPU_STATUS_OK:                  return "OK";
        case SZGPU_STATUS_INVALID_ARGUMENT:    return "invalid argument";
        case SZGPU_STATUS_OUT_OF_MEMORY:       return "out of memory";
        case SZGPU_STATUS_WORKSPACE_TOO_SMALL: return "workspace too small";
        case SZGPU_STATUS_CUDA_ERROR:          return "CUDA error";
        case SZGPU_STATUS_TOO_LARGE:           return "compressed stream too large";
    }
    return "unknown";
}

extern "C" size_t szgpu_num_tiles(size_t n) {
    return (n + SZGPU_TILE_SIZE - 1) / SZGPU_TILE_SIZE;
}

extern "C" size_t szgpu_max_compressed_bytes(size_t n) {
    return szgpu_num_tiles(n) * SZGPU_MAX_TILE_BYTES;
}

extern "C" szgpu_compress_result_t szgpu_make_result(size_t n, size_t cmp_size) {
    size_t numTiles = szgpu_num_tiles(n);
    szgpu_compress_result_t r;
    r.cmp_size    = cmp_size;
    r.num_tiles   = numTiles;
    r.data_offset = numTiles * SZGPU_BLKS_PER_TILE;
    return r;
}

extern "C" szgpu_status_t szgpu_workspace_create(szgpu_workspace_t** out_ws, size_t max_n) {
    if (!out_ws) return SZGPU_STATUS_INVALID_ARGUMENT;
    *out_ws = nullptr;
    if (max_n == 0) return SZGPU_STATUS_INVALID_ARGUMENT;

    szgpu_workspace* w = new(std::nothrow) szgpu_workspace();
    if (!w) return SZGPU_STATUS_OUT_OF_MEMORY;

    size_t numTiles = szgpu_num_tiles(max_n);
    w->max_n_elements = max_n;
    w->gsize_max    = szgpu_grid_size(numTiles);
    w->gsize_sm_max = szgpu_grid_size_sm(numTiles);

    int maxgs = w->gsize_max > w->gsize_sm_max ? w->gsize_max : w->gsize_sm_max;
    w->cmpOffSize = maxgs + 1;

    cudaError_t e = cudaMalloc(&w->d_workspace, sizeof(unsigned int) * w->cmpOffSize * 3);
    if (e != cudaSuccess) { delete w; return SZGPU_STATUS_CUDA_ERROR; }

    w->cmpOffset = w->d_workspace;
    w->locOffset = w->d_workspace + w->cmpOffSize;
    w->flag      = (int*)(w->d_workspace + 2 * w->cmpOffSize);

    e = cudaHostAlloc(&w->h_cmpSize, sizeof(unsigned int), cudaHostAllocDefault);
    if (e != cudaSuccess) {
        cudaFree(w->d_workspace);
        delete w;
        return SZGPU_STATUS_CUDA_ERROR;
    }

    *out_ws = w;
    return SZGPU_STATUS_OK;
}

extern "C" void szgpu_workspace_destroy(szgpu_workspace_t* ws) {
    if (!ws) return;
    if (ws->d_workspace) cudaFree(ws->d_workspace);
    if (ws->h_cmpSize)   cudaFreeHost(ws->h_cmpSize);
    delete ws;
}

extern "C" size_t szgpu_workspace_capacity(const szgpu_workspace_t* ws) {
    return ws ? ws->max_n_elements : 0;
}

template <typename FpT>
static szgpu_status_t szgpu_compress_with_ws(
    const FpT* d_in,
    unsigned char* d_cmp,
    size_t n,
    FpT eb,
    szgpu_workspace_t* ws,
    cudaStream_t stream,
    szgpu_compress_result_t* out_result)
{
    if (!d_in || !d_cmp || !ws || !out_result) return SZGPU_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return SZGPU_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return SZGPU_STATUS_INVALID_ARGUMENT;
    if (n > ws->max_n_elements) return SZGPU_STATUS_WORKSPACE_TOO_SMALL;

    size_t numTiles  = szgpu_num_tiles(n);
    size_t numBlocks = numTiles * SZGPU_BLKS_PER_TILE;
    size_t dataOfs   = numBlocks;

    bool use_small = (n <= SZGPU_SMALL_THRESHOLD);
    int gs = use_small ? szgpu_grid_size_sm(numTiles) : szgpu_grid_size(numTiles);

    cudaError_t e = cudaMemsetAsync(ws->flag, 0, sizeof(int) * (gs + 1), stream);
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    cudaGetLastError();

    if (use_small) {
        szgpu_compress_kernel_T<SZGPU_TILES_PER_THR_SM, FpT><<<gs, SZGPU_TBLOCK_SZ, 8, stream>>>(
            d_in, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, numTiles, dataOfs);
    } else {
        szgpu_compress_kernel_T<SZGPU_TILES_PER_THR, FpT><<<gs, SZGPU_TBLOCK_SZ, 8, stream>>>(
            d_in, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, numTiles, dataOfs);
    }
    if (cudaGetLastError() != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    const unsigned int* d_total = (gs == 1) ? (ws->locOffset + 1)
                                            : (ws->cmpOffset + gs);
    e = cudaMemcpyAsync(ws->h_cmpSize, d_total,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    if ((unsigned long long)numTiles * (SZGPU_MAX_TILE_BYTES - SZGPU_BLKS_PER_TILE)
            > 0xffffffffull) {
        std::vector<unsigned int> parts((size_t)gs);
        e = cudaMemcpy(parts.data(), ws->locOffset + 1,
                       sizeof(unsigned int) * (size_t)gs, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;
        unsigned long long total = 0;
        for (int i = 0; i < gs; ++i) total += parts[(size_t)i];
        if (total > 0xffffffffull) return SZGPU_STATUS_TOO_LARGE;
    }

    out_result->cmp_size    = dataOfs + *ws->h_cmpSize;
    out_result->num_tiles   = numTiles;
    out_result->data_offset = dataOfs;
    return SZGPU_STATUS_OK;
}

template <typename FpT>
static szgpu_status_t szgpu_compress_core(
    const FpT* d_in,
    unsigned char* d_cmp,
    size_t n,
    FpT eb,
    szgpu_workspace_t* ws,
    cudaStream_t stream,
    szgpu_compress_result_t* out_result)
{
    if (ws) return szgpu_compress_with_ws(d_in, d_cmp, n, eb, ws, stream, out_result);
    if (n == 0) return SZGPU_STATUS_INVALID_ARGUMENT;

    szgpu_workspace_t* tmp = nullptr;
    szgpu_status_t s = szgpu_workspace_create(&tmp, n);
    if (s != SZGPU_STATUS_OK) return s;
    s = szgpu_compress_with_ws(d_in, d_cmp, n, eb, tmp, stream, out_result);
    szgpu_workspace_destroy(tmp);
    return s;
}

extern "C" szgpu_status_t szgpu_compress(
    const float* d_in, unsigned char* d_cmp, size_t n, float eb,
    szgpu_workspace_t* ws, cudaStream_t stream, szgpu_compress_result_t* out_result)
{
    return szgpu_compress_core<float>(d_in, d_cmp, n, eb, ws, stream, out_result);
}

extern "C" szgpu_status_t szgpu_compress_f64(
    const double* d_in, unsigned char* d_cmp, size_t n, double eb,
    szgpu_workspace_t* ws, cudaStream_t stream, szgpu_compress_result_t* out_result)
{
    return szgpu_compress_core<double>(d_in, d_cmp, n, eb, ws, stream, out_result);
}

template <typename FpT>
static szgpu_status_t szgpu_decompress_with_ws(
    FpT* d_out,
    const unsigned char* d_cmp,
    size_t n,
    FpT eb,
    const szgpu_compress_result_t* result,
    szgpu_workspace_t* ws,
    cudaStream_t stream)
{
    if (!d_out || !d_cmp || !ws || !result) return SZGPU_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return SZGPU_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return SZGPU_STATUS_INVALID_ARGUMENT;
    if (n > ws->max_n_elements) return SZGPU_STATUS_WORKSPACE_TOO_SMALL;

    const size_t expect_tiles = szgpu_num_tiles(n);
    if (result->num_tiles != expect_tiles) return SZGPU_STATUS_INVALID_ARGUMENT;
    if (result->data_offset != expect_tiles * SZGPU_BLKS_PER_TILE)
        return SZGPU_STATUS_INVALID_ARGUMENT;

    bool use_small = (n <= SZGPU_SMALL_THRESHOLD);
    int gs = use_small ? szgpu_grid_size_sm(result->num_tiles)
                       : szgpu_grid_size(result->num_tiles);

    cudaError_t e = cudaMemsetAsync(ws->flag, 0, sizeof(int) * (gs + 1), stream);
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    size_t origSize = n * sizeof(FpT);
    bool high_cr = (result->cmp_size > 0) && (origSize / result->cmp_size > 100);
    if (high_cr) {
        e = cudaMemsetAsync(d_out, 0, origSize, stream);
        if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;
    }

    cudaGetLastError();

    if (use_small) {
        szgpu_decompress_kernel_T<SZGPU_TILES_PER_THR_SM, FpT><<<gs, SZGPU_TBLOCK_SZ, 8, stream>>>(
            d_out, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, result->num_tiles, result->data_offset, high_cr);
    } else {
        szgpu_decompress_kernel_T<SZGPU_TILES_PER_THR, FpT><<<gs, SZGPU_TBLOCK_SZ, 8, stream>>>(
            d_out, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, result->num_tiles, result->data_offset, high_cr);
    }
    if (cudaGetLastError() != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;
    return SZGPU_STATUS_OK;
}

template <typename FpT>
static szgpu_status_t szgpu_decompress_core(
    FpT* d_out,
    const unsigned char* d_cmp,
    size_t n,
    FpT eb,
    const szgpu_compress_result_t* result,
    szgpu_workspace_t* ws,
    cudaStream_t stream)
{
    if (ws) return szgpu_decompress_with_ws(d_out, d_cmp, n, eb, result, ws, stream);
    if (n == 0) return SZGPU_STATUS_INVALID_ARGUMENT;

    szgpu_workspace_t* tmp = nullptr;
    szgpu_status_t s = szgpu_workspace_create(&tmp, n);
    if (s != SZGPU_STATUS_OK) return s;
    s = szgpu_decompress_with_ws(d_out, d_cmp, n, eb, result, tmp, stream);
    cudaError_t e = cudaStreamSynchronize(stream);
    szgpu_workspace_destroy(tmp);
    if (s == SZGPU_STATUS_OK && e != cudaSuccess) s = SZGPU_STATUS_CUDA_ERROR;
    return s;
}

extern "C" szgpu_status_t szgpu_decompress(
    float* d_out, const unsigned char* d_cmp, size_t n, float eb,
    const szgpu_compress_result_t* result, szgpu_workspace_t* ws, cudaStream_t stream)
{
    return szgpu_decompress_core<float>(d_out, d_cmp, n, eb, result, ws, stream);
}

extern "C" szgpu_status_t szgpu_decompress_f64(
    double* d_out, const unsigned char* d_cmp, size_t n, double eb,
    const szgpu_compress_result_t* result, szgpu_workspace_t* ws, cudaStream_t stream)
{
    return szgpu_decompress_core<double>(d_out, d_cmp, n, eb, result, ws, stream);
}

template <typename FpT>
static szgpu_status_t szgpu_compress_hostptr_core(
    const FpT* in,
    unsigned char* cmp,
    size_t n,
    FpT eb,
    size_t* out_cmp_size)
{
    if (!in || !cmp || !out_cmp_size) return SZGPU_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return SZGPU_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return SZGPU_STATUS_INVALID_ARGUMENT;

    *out_cmp_size = 0;

    FpT* d_in = nullptr;
    unsigned char* d_cmp = nullptr;
    szgpu_workspace_t* ws = nullptr;

    cudaError_t e = cudaMalloc(&d_in, n * sizeof(FpT));
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    e = cudaMalloc(&d_cmp, szgpu_max_compressed_bytes(n));
    if (e != cudaSuccess) {
        szgpu_hostptr_release(d_in, nullptr, nullptr);
        return SZGPU_STATUS_CUDA_ERROR;
    }

    szgpu_status_t s = szgpu_workspace_create(&ws, n);
    if (s != SZGPU_STATUS_OK) {
        szgpu_hostptr_release(d_in, d_cmp, nullptr);
        return s;
    }

    e = cudaMemcpy(d_in, in, n * sizeof(FpT), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        szgpu_hostptr_release(d_in, d_cmp, ws);
        return SZGPU_STATUS_CUDA_ERROR;
    }

    szgpu_compress_result_t r;
    s = szgpu_compress_core<FpT>(d_in, d_cmp, n, eb, ws, 0, &r);
    if (s != SZGPU_STATUS_OK) {
        szgpu_hostptr_release(d_in, d_cmp, ws);
        return s;
    }

    e = cudaMemcpy(cmp, d_cmp, r.cmp_size, cudaMemcpyDeviceToHost);
    szgpu_hostptr_release(d_in, d_cmp, ws);
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    *out_cmp_size = r.cmp_size;
    return SZGPU_STATUS_OK;
}

extern "C" szgpu_status_t szgpu_compress_hostptr(
    const float* in, unsigned char* cmp, size_t n, float eb, size_t* out_cmp_size)
{
    return szgpu_compress_hostptr_core<float>(in, cmp, n, eb, out_cmp_size);
}

extern "C" szgpu_status_t szgpu_compress_hostptr_f64(
    const double* in, unsigned char* cmp, size_t n, double eb, size_t* out_cmp_size)
{
    return szgpu_compress_hostptr_core<double>(in, cmp, n, eb, out_cmp_size);
}

template <typename FpT>
static szgpu_status_t szgpu_decompress_hostptr_core(
    FpT* out,
    const unsigned char* cmp,
    size_t cmp_size,
    size_t n,
    FpT eb)
{
    if (!out || !cmp)  return SZGPU_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return SZGPU_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return SZGPU_STATUS_INVALID_ARGUMENT;

    size_t header = szgpu_num_tiles(n) * SZGPU_BLKS_PER_TILE;
    if (cmp_size == 0 || cmp_size < header) return SZGPU_STATUS_INVALID_ARGUMENT;

    FpT* d_out = nullptr;
    unsigned char* d_cmp = nullptr;
    szgpu_workspace_t* ws = nullptr;

    cudaError_t e = cudaMalloc(&d_out, n * sizeof(FpT));
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    e = cudaMalloc(&d_cmp, cmp_size);
    if (e != cudaSuccess) {
        szgpu_hostptr_release(d_out, nullptr, nullptr);
        return SZGPU_STATUS_CUDA_ERROR;
    }

    szgpu_status_t s = szgpu_workspace_create(&ws, n);
    if (s != SZGPU_STATUS_OK) {
        szgpu_hostptr_release(d_out, d_cmp, nullptr);
        return s;
    }

    e = cudaMemcpy(d_cmp, cmp, cmp_size, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        szgpu_hostptr_release(d_out, d_cmp, ws);
        return SZGPU_STATUS_CUDA_ERROR;
    }

    szgpu_compress_result_t r = szgpu_make_result(n, cmp_size);
    s = szgpu_decompress_core<FpT>(d_out, d_cmp, n, eb, &r, ws, 0);
    if (s != SZGPU_STATUS_OK) {
        szgpu_hostptr_release(d_out, d_cmp, ws);
        return s;
    }

    e = cudaStreamSynchronize(0);
    if (e != cudaSuccess) {
        szgpu_hostptr_release(d_out, d_cmp, ws);
        return SZGPU_STATUS_CUDA_ERROR;
    }

    e = cudaMemcpy(out, d_out, n * sizeof(FpT), cudaMemcpyDeviceToHost);
    szgpu_hostptr_release(d_out, d_cmp, ws);
    if (e != cudaSuccess) return SZGPU_STATUS_CUDA_ERROR;

    return SZGPU_STATUS_OK;
}

extern "C" szgpu_status_t szgpu_decompress_hostptr(
    float* out, const unsigned char* cmp, size_t cmp_size, size_t n, float eb)
{
    return szgpu_decompress_hostptr_core<float>(out, cmp, cmp_size, n, eb);
}

extern "C" szgpu_status_t szgpu_decompress_hostptr_f64(
    double* out, const unsigned char* cmp, size_t cmp_size, size_t n, double eb)
{
    return szgpu_decompress_hostptr_core<double>(out, cmp, cmp_size, n, eb);
}
