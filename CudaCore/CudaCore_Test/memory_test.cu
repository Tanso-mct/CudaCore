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

TEST(CudaCore, ManagedMemory)
{
    // Allocate managed memory
    void* mArray = nullptr;
    size_t size = 1024 * sizeof(int);
    CudaCore::MallocManaged(&mArray, size);

    // Free managed memory
    CudaCore::FreeManaged(&mArray);

    EXPECT_EQ(mArray, nullptr);
}

TEST(CudaCore, ManagedMemoryTryVer)
{
    bool result = true;

    // Allocate managed memory with error checking
    void* mArray = nullptr;
    size_t size = 1024 * sizeof(int);
    result = CudaCore::TryMallocManaged(&mArray, size);
    ASSERT_EQ(result, true);

    // Free managed memory
    result = CudaCore::TryFreeManaged(&mArray);
    ASSERT_EQ(result, true);

    EXPECT_EQ(mArray, nullptr);
}

TEST(CudaCore, ArrayMemory)
{
    // Allocate array memory
    cudaArray_t array = nullptr;
    size_t width = 1024;
    size_t height = 1024;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CudaCore::MallocArray(&array, &channelDesc, width, height, cudaArrayDefault);

    // Free array memory
    CudaCore::FreeArray(&array);

    EXPECT_EQ(array, nullptr);
}

TEST(CudaCore, ArrayMemoryTryVer)
{
    bool result = true;

    // Allocate array memory with error checking
    cudaArray_t array = nullptr;
    size_t width = 1024;
    size_t height = 1024;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    result = CudaCore::TryMallocArray(&array, &channelDesc, width, height, cudaArrayDefault);
    ASSERT_EQ(result, true);

    // Free array memory
    result = CudaCore::TryFreeArray(&array);
    ASSERT_EQ(result, true);

    EXPECT_EQ(array, nullptr);
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

TEST(CudaCore, SurfaceObject)
{
    // Allocate array memory for surface object 
    cudaArray_t array = nullptr;
    size_t width = 1024;
    size_t height = 1024;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CudaCore::MallocArray(&array, &channelDesc, width, height, cudaArrayDefault);

    // Create surface object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surfaceObj = 0;
    CudaCore::CreateSurfaceObj(&surfaceObj, &resDesc);

    // Destroy surface object
    CudaCore::DestroySurfaceObj(&surfaceObj);

    // Free array memory
    CudaCore::FreeArray(&array);

    EXPECT_EQ(array, nullptr);
    EXPECT_EQ(surfaceObj, 0);
}

TEST(CudaCore, SurfaceObjectTryVer)
{
    bool result = true;

    // Allocate array memory for surface object with error checking
    cudaArray_t array = nullptr;
    size_t width = 1024;
    size_t height = 1024;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    result = CudaCore::TryMallocArray(&array, &channelDesc, width, height, cudaArrayDefault);
    ASSERT_EQ(result, true);

    // Create surface object with error checking
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surfaceObj = 0;
    result = CudaCore::TryCreateSurfaceObj(&surfaceObj, &resDesc);
    ASSERT_EQ(result, true);

    // Destroy surface object with error checking
    result = CudaCore::TryDestroySurfaceObj(&surfaceObj);
    ASSERT_EQ(result, true);

    // Free array memory with error checking
    result = CudaCore::TryFreeArray(&array);
    ASSERT_EQ(result, true);

    EXPECT_EQ(array, nullptr);
    EXPECT_EQ(surfaceObj, 0);
}
TEST(CudaCore, TextureObject)
{
    // Allocate array memory for texture object
    cudaArray_t array = nullptr;
    size_t width = 1024;
    size_t height = 1024;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CudaCore::MallocArray(&array, &channelDesc, width, height, cudaArrayDefault);

    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureObject_t texObj = 0;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    CudaCore::CreateTextureObj(&texObj, &resDesc, &texDesc, nullptr);

    // Destroy texture object
    CudaCore::DestroyTextureObj(&texObj);

    // Free array memory
    CudaCore::FreeArray(&array);

    EXPECT_EQ(array, nullptr);
    EXPECT_EQ(texObj, 0);
}

TEST(CudaCore, TextureObjectTryVer)
{
    bool result = true;

    // Allocate array memory for texture object with error checking
    cudaArray_t array = nullptr;
    size_t width = 1024;
    size_t height = 1024;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    result = CudaCore::TryMallocArray(&array, &channelDesc, width, height, cudaArrayDefault);
    ASSERT_EQ(result, true);

    // Create texture object with error checking
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureObject_t texObj = 0;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    result = CudaCore::TryCreateTextureObj(&texObj, &resDesc, &texDesc, nullptr);
    ASSERT_EQ(result, true);

    // Destroy texture object with error checking
    result = CudaCore::TryDestroyTextureObj(&texObj);
    ASSERT_EQ(result, true);

    // Free array memory with error checking
    result = CudaCore::TryFreeArray(&array);
    ASSERT_EQ(result, true);

    EXPECT_EQ(array, nullptr);
    EXPECT_EQ(texObj, 0);
}