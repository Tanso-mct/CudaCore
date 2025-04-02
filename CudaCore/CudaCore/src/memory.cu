#include "CudaCore/include/pch.h"
#include "CudaCore/include/memory.cuh"

#include "CudaCore/include/console.cuh"

CUDA_CORE bool CudaCore::CheckCudaErr(cudaError_t call)
{
    if (call != cudaSuccess) return false;
    return true;
}

namespace
{

std::string GetPointerAddress(const void *ptr)
{
    std::stringstream ss;
    ss << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << reinterpret_cast<uintptr_t>(ptr);
    return ss.str();
}

}

CUDA_CORE void CudaCore::Malloc(void **devPtr, size_t size)
{
    if (CheckCudaErr(cudaMalloc(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA malloc succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
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
    if (CheckCudaErr(cudaMalloc(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA malloc succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
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

CUDA_CORE void CudaCore::Free(void **devPtr)
{
    if (CheckCudaErr(cudaFree(*devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA free succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
        });
#endif
        *devPtr = nullptr;
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
CUDA_CORE bool CudaCore::TryFree(void **devPtr)
{
    if (CheckCudaErr(cudaFree(*devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA free succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
        });
#endif
        *devPtr = nullptr;
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
    if (CheckCudaErr(cudaMallocHost(ptr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
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
    if (CheckCudaErr(cudaMallocHost(ptr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
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

CUDA_CORE void CudaCore::FreeHost(void **ptr)
{
    if (CheckCudaErr(cudaFreeHost(*ptr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
        });
#endif
        *ptr = nullptr;
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA freeHost failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        throw std::runtime_error("CudaCore::FreeHost failed");
    }
}

CUDA_CORE bool CudaCore::TryFreeHost(void **ptr)
{
    if (CheckCudaErr(cudaFreeHost(*ptr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
        });
#endif
        *ptr = nullptr;
        return true;
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA freeHost failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::Memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
    if (CheckCudaErr(cudaMemcpy(dst, src, size, kind)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memcpy succeeded.", 
            "dst address:" + ::GetPointerAddress(dst),
            "src address:" + ::GetPointerAddress(src),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA memcpy failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        throw std::runtime_error("CudaCore::Memcpy failed");
    }
}

CUDA_CORE bool CudaCore::TryMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
    if (CheckCudaErr(cudaMemcpy(dst, src, size, kind)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memcpy succeeded.", 
            "dst address:" + ::GetPointerAddress(dst),
            "src address:" + ::GetPointerAddress(src),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA memcpy failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::Memset(void *devPtr, int value, size_t size)
{
    if (CheckCudaErr(cudaMemset(devPtr, value, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memset succeeded.", 
            "address:" + ::GetPointerAddress(devPtr),
            "value:" + std::to_string(value),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA memset failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        throw std::runtime_error("CudaCore::Memset failed");
    }
}

CUDA_CORE bool CudaCore::TryMemset(void *devPtr, int value, size_t size)
{
    if (CheckCudaErr(cudaMemset(devPtr, value, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memset succeeded.", 
            "address:" + ::GetPointerAddress(devPtr),
            "value:" + std::to_string(value),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        CudaCore::CoutErr
        ({
            "CUDA memset failed.", 
            "code:" + std::to_string(cudaGetLastError()),
            "reason:" + std::string(cudaGetErrorString(cudaGetLastError()))
        });

        return false;
    }
}
