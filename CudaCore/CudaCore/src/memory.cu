#include "CudaCore/include/pch.h"
#include "CudaCore/include/memory.cuh"

#include "CudaCore/include/funcs.cuh"

CUDA_CORE bool CudaCore::CheckCuda(cudaError_t call)
{
    if (call != cudaSuccess) return false;
    return true;
}

CUDA_CORE void CudaCore::Malloc(void **devPtr, size_t size)
{
    if (CheckCuda(cudaMalloc(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA malloc succeeded.", 
            "address:" + std::to_string(reinterpret_cast<uintptr_t>(*devPtr)),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA malloc failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        throw std::runtime_error("CudaCore::Malloc failed");
    }
}

CUDA_CORE bool CudaCore::TryMalloc(void **devPtr, size_t size)
{
    if (CheckCuda(cudaMalloc(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA malloc succeeded.", 
            "address:" + std::to_string(reinterpret_cast<uintptr_t>(*devPtr)),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA malloc failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::Free(void *devPtr)
{
    if (CheckCuda(cudaFree(devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA free succeeded.", 
            "address:" + std::to_string(reinterpret_cast<uintptr_t>(devPtr)),
        });
#endif
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA free failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        throw std::runtime_error("CudaCore::Free failed");
    }
}
CUDA_CORE bool CudaCore::TryFree(void *devPtr)
{
    if (CheckCuda(cudaFree(devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA free succeeded.", 
            "address:" + std::to_string(reinterpret_cast<uintptr_t>(devPtr)),
        });
#endif
        return true;
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA free failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::MallocHost(void **ptr, size_t size)
{
    if (CheckCuda(cudaMallocHost(ptr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocHost succeeded.", 
            "address:" + std::to_string(reinterpret_cast<uintptr_t>(*ptr)),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA mallocHost failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        throw std::runtime_error("CudaCore::MallocHost failed");
    }
}

CUDA_CORE bool CudaCore::TryMallocHost(void **ptr, size_t size)
{
    if (CheckCuda(cudaMallocHost(ptr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocHost succeeded.", 
            "address:" + std::to_string(reinterpret_cast<uintptr_t>(*ptr)),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA mallocHost failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::FreeHost(void *ptr)
{
    
}

CUDA_CORE bool CudaCore::TryFreeHost(void *ptr)
{
    return true;
}

CUDA_CORE void CudaCore::Memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
    
}

CUDA_CORE bool CudaCore::TryMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
    return true;
}

CUDA_CORE void CudaCore::Memset(void *devPtr, int value, size_t size)
{
    
}

CUDA_CORE bool CudaCore::TryMemset(void *devPtr, int value, size_t size)
{
    return true;
}
