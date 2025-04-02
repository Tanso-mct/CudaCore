#pragma once

#include "CudaCore/include/config.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CudaCore
{

CUDA_CORE bool CheckCudaErr(cudaError_t call);

CUDA_CORE void Malloc(void **devPtr, size_t size);
CUDA_CORE bool TryMalloc(void **devPtr, size_t size);
CUDA_CORE void Free(void **devPtr);
CUDA_CORE bool TryFree(void **devPtr);

CUDA_CORE void MallocHost(void **ptr, size_t size);
CUDA_CORE bool TryMallocHost(void **ptr, size_t size);
CUDA_CORE void FreeHost(void **ptr);
CUDA_CORE bool TryFreeHost(void **ptr);

CUDA_CORE void Memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind);
CUDA_CORE bool TryMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind);

CUDA_CORE void Memset(void *devPtr, int value, size_t size);
CUDA_CORE bool TryMemset(void *devPtr, int value, size_t size);



} // namespace CudaCore