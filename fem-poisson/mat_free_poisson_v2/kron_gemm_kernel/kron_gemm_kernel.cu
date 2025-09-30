#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;


#ifndef N
#define N 3
#endif



using dtype = float;


// CUDA kernel: 
/**
 * All row major
 * A [N x N], B [eN x N^3], C [eN x N^3]    ; eN = element count
 * 
 * CUDA kernel for
 *      if direction == 0
 *          C[i] = (I ⊗ I ⊗ A) x B[i]
 *      else if direction == 1
 *          C[i] = (I ⊗ A ⊗ I) x B[i]
 *      else if direction == 2
 *          C[i] = (A ⊗ I ⊗ I) x B[i]
 * 
 * 
 */

constexpr int block_size() {
    static_assert(N <= 10);
    switch (N) {
        case 2: return 256;
        case 3: return 288;
        case 4: return 256;
        case 5: return 128;
        case 6: return 288;
        case 7: return 448;
        case 8: return 256;
        case 9: return 256;
        case 10: return 416;
        default: return -1;
    }
}


template <int direction>
__global__ void kron_gemm_kernel(
    const dtype* __restrict__ A,    // [N x N]
    const dtype* __restrict__ b,    // [eN x N^3]
    dtype* __restrict__ c,          // [eN x N^3]
    size_t eN                       // no of elements
) {
    static_assert(direction == 0 || direction == 1 || direction == 2, 
                  "Invalid direction: Only 0, 1, 2 are allowed");
    
    constexpr int tileN = block_size();
    static_assert((N*N) <= tileN);
    constexpr int eN_tile = tileN / (N*N) ;     // no of elements per tile

    const int tid_start = blockIdx.x * tileN;
    const int tx = threadIdx.x;

    const int B_len = eN * N * N * N;
    const int C_len = B_len;

    const int B_tile_start = blockIdx.x * eN_tile * N * N * N;
    const int C_tile_start = B_tile_start;





    // Shared memory for A, B tile, C tile
    __shared__ dtype Ash[N][N];
    __shared__ dtype Bsh[N][eN_tile * N * N];
    __shared__ dtype Csh[N][eN_tile * N * N];


    // Load A into shared memory (coalesced, first N*N threads)
    if (tx < N * N) {
        int mi = tx / N;
        int ki = tx % N;
        Ash[mi][ki] = A[mi * N + ki];
    }

    // Load B tile into shared memory (each thread loads at most N elements)

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const int tile_idx = tileN*i + tx;
        const int B_idx = B_tile_start + tile_idx;

        int col_idx, row_idx;

        if constexpr (direction == 0) {row_idx = tile_idx % N; col_idx = tile_idx / N;}
        else if constexpr (direction == 1) {row_idx = (tile_idx / N) % N; col_idx = ((tile_idx / (N*N)) * N) + (tile_idx % N);}
        else if constexpr (direction == 2) {row_idx = (tile_idx / (N*N)) % N; col_idx = ((tile_idx / (N*N*N)) * N * N) + (tile_idx % (N*N));}

        if (tile_idx < eN_tile * N * N *N && B_idx < B_len)
        {
            Bsh[row_idx][col_idx] = b[B_idx];
        }           
        
    }

    __syncthreads();

    if (tx < eN_tile * N * N) {
        #pragma unroll
        for (int mi = 0; mi < N; ++mi) {
            dtype sum = 0;
            #pragma unroll
            for (int ki = 0; ki < N; ++ki) {
                sum += Ash[mi][ki] * Bsh[ki][tx];
            }
            Csh[mi][tx] = sum;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const int tile_idx = tileN*i + tx;
        const int C_idx = C_tile_start + tile_idx;
        int col_idx, row_idx;

        if constexpr (direction == 0) {row_idx = tile_idx % N; col_idx = tile_idx / N;}
        else if constexpr (direction == 1) {row_idx = (tile_idx / N) % N; col_idx = ((tile_idx / (N*N)) * N) + (tile_idx % N);}
        else if constexpr (direction == 2) {row_idx = (tile_idx / (N*N)) % N; col_idx = ((tile_idx / (N*N*N)) * N * N) + (tile_idx % (N*N));}

        if (tile_idx < eN_tile * N * N *N && C_idx < C_len)
        {
            c[C_idx] = Csh[row_idx][col_idx];
        }           
        
    }



}

template <int direction>
ffi::Error kron_gemm_kernel_host(cudaStream_t stream, ffi::Buffer<ffi::F32> A,
                      ffi::Buffer<ffi::F32> b,
                      ffi::ResultBuffer<ffi::F32> result_c, size_t eN)
{
    constexpr int tileN = block_size();
    constexpr int eN_tile = tileN / (N*N) ;

    int nblocks = (eN + eN_tile - 1) / eN_tile;
    kron_gemm_kernel<direction><<<nblocks, tileN, /*shared_mem=*/0, stream>>>(
        A.typed_data(), b.typed_data(), result_c->typed_data(),
        eN);
    // Check for launch time errors. Note that this function may also
    // return error codes from previous, asynchronous launches. This
    // means that an error status returned here could have been caused
    // by a different kernel previously launched by XLA.
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess)
    {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IIA_kernel, kron_gemm_kernel_host<0>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::F32>>()             // a
        .Arg<ffi::Buffer<ffi::F32>>()             // b
        .Ret<ffi::Buffer<ffi::F32>>()             // result_c
        .Attr<size_t>("eN"),
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    IAI_kernel, kron_gemm_kernel_host<1>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::F32>>()             // a
        .Arg<ffi::Buffer<ffi::F32>>()             // b
        .Ret<ffi::Buffer<ffi::F32>>()             // result_c
        .Attr<size_t>("eN"),
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AII_kernel, kron_gemm_kernel_host<2>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::F32>>()             // a
        .Arg<ffi::Buffer<ffi::F32>>()             // b
        .Ret<ffi::Buffer<ffi::F32>>()             // result_c
        .Attr<size_t>("eN"),
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled