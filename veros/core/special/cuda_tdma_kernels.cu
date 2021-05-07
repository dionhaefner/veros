#include <array>
#include <cstddef>

#include <cuda_runtime.h>

#include "cuda_tdma_kernels.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename DTYPE>
__global__ void TridiagKernel(
    const DTYPE *a,
    const DTYPE *b,
    const DTYPE *c,
    const DTYPE *d,
    DTYPE *cp,
    DTYPE *dp,
    const int num_systems,
    const int system_depth
){
  // TDMA algorithm
  // Solution is written to dp
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_systems;
       idx += blockDim.x * gridDim.x) {

    if (idx >= system_depth * num_systems) {
      return;
    }

    int indj = idx;

    DTYPE denom;
    DTYPE ai;

    DTYPE b0 = b[indj];
    DTYPE cm1 = c[indj] / b0;
    DTYPE dm1 = d[indj] / b0;

    cp[indj] = cm1;
    dp[indj] = dm1;

    // forward pass
    for (int j = 0; j < system_depth-1; ++j) {
      indj += num_systems;
      ai = a[indj];
      denom = 1.0f / (b[indj] - ai * cm1);

      cm1 = c[indj] * denom;
      dm1 = (d[indj] - ai * dm1) * denom;

      cp[indj] = cm1;
      dp[indj] = dm1;
    }

    // backward pass
    for (int j = 0; j < system_depth-1; ++j) {
      indj -= num_systems;
      dp[indj] -= cp[indj] * dp[indj + num_systems];
    }
  }
}

// Unpacks a descriptor object from a byte string.
template <typename T>
const T* UnpackDescriptor(const char* opaque, size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Descriptor was not encoded correctly");
  }
  return reinterpret_cast<const T*>(opaque);
}

template <typename DTYPE>
void CudaTridiag(cudaStream_t stream, void** buffers, const char* opaque,
                 size_t opaque_len) {

  const auto& descriptor = *UnpackDescriptor<TridiagDescriptor>(opaque, opaque_len);
  const int num_systems = descriptor.num_systems;
  const int system_depth = descriptor.system_depth;

  const DTYPE* a = reinterpret_cast<const DTYPE*>(buffers[0]);
  const DTYPE* b = reinterpret_cast<const DTYPE*>(buffers[1]);
  const DTYPE* c = reinterpret_cast<const DTYPE*>(buffers[2]);
  const DTYPE* d = reinterpret_cast<const DTYPE*>(buffers[3]);

  DTYPE* out = reinterpret_cast<DTYPE*>(buffers[4]);
  DTYPE* workspace = reinterpret_cast<DTYPE*>(buffers[5]);

  const int BLOCK_SIZE = 128;
  const int grid_dim = std::min<int>(1024, (num_systems + BLOCK_SIZE - 1) / BLOCK_SIZE);
  TridiagKernel<DTYPE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(a, b, c, d, workspace, out, num_systems, system_depth);
  gpuErrchk(cudaPeekAtLastError());
}

void CudaTridiagFloat(cudaStream_t stream, void** buffers, const char* opaque,
                      size_t opaque_len) {
  CudaTridiag<float>(stream, buffers, opaque, opaque_len);
}

void CudaTridiagDouble(cudaStream_t stream, void** buffers, const char* opaque,
                      size_t opaque_len) {
  CudaTridiag<double>(stream, buffers, opaque, opaque_len);
}
