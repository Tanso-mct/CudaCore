#include "pch.h"

TEST(CudaCore, DeviceMemory) 
{
    // Allocate device memory
    void* dArray = nullptr;
    size_t size = 1024 * sizeof(int);
    CudaCore::Malloc(&dArray, size);

    // Free device memory
    CudaCore::Free(&dArray);

    EXPECT_EQ(dArray, nullptr);
}

TEST(CudaCore, DeviceMemoryTryVer)
{
    bool result = true;

    // Allocate device memory with error checking
    void* dArray = nullptr;
    size_t size = 1024 * sizeof(int);
    result = CudaCore::TryMalloc(&dArray, size);
    ASSERT_EQ(result, true);

    // Free device memory
    result = CudaCore::TryFree(&dArray);
    ASSERT_EQ(result, true);

    EXPECT_EQ(dArray, nullptr);
}

TEST(CudaCore, HostMemory) 
{
    // Allocate host memory
    void* hArray = nullptr;
    size_t size = 1024 * sizeof(int);
    CudaCore::MallocHost(&hArray, size);

    // Free host memory
    CudaCore::FreeHost(&hArray);

    EXPECT_EQ(hArray, nullptr);
}

TEST(CudaCore, HostMemoryTryVer) 
{
    bool result = true;

    // Allocate host memory with error checking
    void* hArray = nullptr;
    size_t size = 1024 * sizeof(int);
    result = CudaCore::TryMallocHost(&hArray, size);
    ASSERT_EQ(result, true);

    // Free host memory
    result = CudaCore::TryFreeHost(&hArray);
    ASSERT_EQ(result, true);

    EXPECT_EQ(hArray, nullptr);
}

TEST(CudaCore, Memcpy)
{
    // Allocate host and device memory
    void* hArray = nullptr;
    void* dArray = nullptr;
    size_t size = 1024 * sizeof(int);
    CudaCore::MallocHost(&hArray, size);
    CudaCore::Malloc(&dArray, size);

    // Initialize host memory
    int* hArrayInt = static_cast<int*>(hArray);
    for (size_t i = 0; i < size / sizeof(int); i++)
    {
        hArrayInt[i] = static_cast<int>(i);
    }

    // Copy data from host to device
    CudaCore::Memcpy(dArray, hArray, size, cudaMemcpyHostToDevice);

    // Copy data from device to host
    CudaCore::Memcpy(hArray, dArray, size, cudaMemcpyDeviceToHost);

    // Free memory
    CudaCore::FreeHost(&hArray);
    CudaCore::Free(&dArray);

    EXPECT_EQ(hArray, nullptr);
    EXPECT_EQ(dArray, nullptr);
}

TEST(CudaCore, MemcpyTryVer)
{
    bool result = true;

    // Allocate host and device memory with error checking
    void* hArray = nullptr;
    void* dArray = nullptr;
    size_t size = 1024 * sizeof(int);
    result = CudaCore::TryMallocHost(&hArray, size);
    ASSERT_EQ(result, true);
    result = CudaCore::TryMalloc(&dArray, size);
    ASSERT_EQ(result, true);

    // Initialize host memory
    int* hArrayInt = static_cast<int*>(hArray);
    for (size_t i = 0; i < size / sizeof(int); i++)
    {
        hArrayInt[i] = static_cast<int>(i);
    }

    // Copy data from host to device with error checking
    result = CudaCore::TryMemcpy(dArray, hArray, size, cudaMemcpyHostToDevice);
    ASSERT_EQ(result, true);

    // Copy data from device to host with error checking
    result = CudaCore::TryMemcpy(hArray, dArray, size, cudaMemcpyDeviceToHost);
    ASSERT_EQ(result, true);

    // Free memory with error checking
    result = CudaCore::TryFreeHost(&hArray);
    ASSERT_EQ(result, true);
    result = CudaCore::TryFree(&dArray);
    ASSERT_EQ(result, true);

    EXPECT_EQ(hArray, nullptr);
    EXPECT_EQ(dArray, nullptr);
}

TEST(CudaCore, DeviceMemset)
{
    // Allocate device memory
    void* dArray = nullptr;
    size_t size = 1024 * sizeof(int);
    CudaCore::Malloc(&dArray, size);

    // Set device memory to zero
    CudaCore::Memset(dArray, 0, size);

    // Free device memory
    CudaCore::Free(&dArray);

    EXPECT_EQ(dArray, nullptr);
}

TEST(CudaCore, DeviceMemsetTryVer)
{
    bool result = true;

    // Allocate device memory with error checking
    void* dArray = nullptr;
    size_t size = 1024 * sizeof(int);
    result = CudaCore::TryMalloc(&dArray, size);
    ASSERT_EQ(result, true);

    // Set device memory to zero with error checking
    result = CudaCore::TryMemset(dArray, 0, size);
    ASSERT_EQ(result, true);

    // Free device memory with error checking
    result = CudaCore::TryFree(&dArray);
    ASSERT_EQ(result, true);

    EXPECT_EQ(dArray, nullptr);
}