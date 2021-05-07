#pragma once

#include <cuda_runtime.h>

struct TridiagDescriptor {
  int num_systems;
  int system_depth;
};

void CudaTridiagFloat(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);
void CudaTridiagDouble(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);
