# cython: language=c++

from cpython.pycapsule cimport PyCapsule_New


cdef extern from "cuda_runtime_api.h":
    ctypedef void* cudaStream_t
    cdef struct TridiagDescriptor:
        int num_systems
        int system_depth


cdef extern from "cuda_tdma_kernels.h":
    void CudaTridiagFloat(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
    void CudaTridiagDouble(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)


cpdef bytes build_tridiag_descriptor(int num_systems, int system_depth):
    cdef TridiagDescriptor desc = TridiagDescriptor(num_systems, system_depth)
    return bytes((<char*> &desc)[:sizeof(TridiagDescriptor)])


gpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    gpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"tdma_cuda_float", <void*>(CudaTridiagFloat))
register_custom_call_target(b"tdma_cuda_double", <void*>(CudaTridiagDouble))
