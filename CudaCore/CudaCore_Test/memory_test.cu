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

TEST(CudaCore, Array3DMemory)
{
    // Allocate 3D array memory
    cudaArray_t array = nullptr;
    size_t width = 256;
    size_t height = 256;
    size_t depth = 256;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CudaCore::Malloc3DArray(&array, &channelDesc, width, height, depth, cudaArrayDefault);

    // Free 3D array memory
    CudaCore::Free3D(&array);

    EXPECT_EQ(array, nullptr);
}

TEST(CudaCore, Array3DMemoryTryVer)
{
    bool result = true;

    // Allocate 3D array memory with error checking
    cudaArray_t array = nullptr;
    size_t width = 256;
    size_t height = 256;
    size_t depth = 256;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    result = CudaCore::TryMalloc3DArray(&array, &channelDesc, width, height, depth, cudaArrayDefault);
    ASSERT_EQ(result, true);

    // Free 3D array memory
    result = CudaCore::TryFree3D(&array);
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

TEST(CudaCore, RegisterResource)
{
    // Declare device and device context pointers
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(float) * 16;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = 0;

    float initData[16] = { 0 };
    D3D11_SUBRESOURCE_DATA initDataDesc = {};
    initDataDesc.pSysMem = initData;

    ID3D11Buffer* buffer = nullptr;
    hr = device->CreateBuffer(&bufferDesc, &initDataDesc, &buffer);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA
    cudaGraphicsResource* cudaResource = nullptr;
    CudaCore::RegisterResource(&cudaResource, buffer, cudaGraphicsRegisterFlagsNone);

    // Unregister the resource
    CudaCore::UnregisterResource(&cudaResource);

    EXPECT_EQ(cudaResource, nullptr);

    buffer->Release();
    device->Release();
    deviceContext->Release();
}

TEST(CudaCore, RegisterResourceTryVer)
{
    bool result = true;

    // Declare device and device context pointers with error checking
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation with error checking
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(float) * 16;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = 0;

    float initData[16] = { 0 };
    D3D11_SUBRESOURCE_DATA initDataDesc = {};
    initDataDesc.pSysMem = initData;

    ID3D11Buffer* buffer = nullptr;
    hr = device->CreateBuffer(&bufferDesc, &initDataDesc, &buffer);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA with error checking
    cudaGraphicsResource* cudaResource = nullptr;
    result = CudaCore::TryRegisterResource(&cudaResource, buffer, cudaGraphicsRegisterFlagsNone);
    ASSERT_EQ(result, true);

    // Unregister the resource with error checking
    result = CudaCore::TryUnregisterResource(&cudaResource);
    ASSERT_EQ(result, true);

    EXPECT_EQ(cudaResource, nullptr);

    buffer->Release();
    device->Release();
    deviceContext->Release();
}

TEST(CudaCore, MapResource)
{
    // Declare device and device context pointers
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(float) * 16;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = 0;

    float initData[16] = { 0 };
    D3D11_SUBRESOURCE_DATA initDataDesc = {};
    initDataDesc.pSysMem = initData;

    ID3D11Buffer* buffer = nullptr;
    hr = device->CreateBuffer(&bufferDesc, &initDataDesc, &buffer);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA
    cudaGraphicsResource* cudaResource = nullptr;
    CudaCore::RegisterResource(&cudaResource, buffer, cudaGraphicsRegisterFlagsNone);

    // Map the resource
    int count = 1;
    CudaCore::MapResource(count, &cudaResource);

    // Unmap the resource
    CudaCore::UnmapResource(count, &cudaResource);

    // Unregister the resource
    CudaCore::UnregisterResource(&cudaResource);

    buffer->Release();
    device->Release();
    deviceContext->Release();
}

TEST(CudaCore, MapResourceTryVer)
{
    bool result = true;

    // Declare device and device context pointers with error checking
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation with error checking
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(float) * 16;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = 0;

    float initData[16] = { 0 };
    D3D11_SUBRESOURCE_DATA initDataDesc = {};
    initDataDesc.pSysMem = initData;

    ID3D11Buffer* buffer = nullptr;
    hr = device->CreateBuffer(&bufferDesc, &initDataDesc, &buffer);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA with error checking
    cudaGraphicsResource* cudaResource = nullptr;
    result = CudaCore::TryRegisterResource(&cudaResource, buffer, cudaGraphicsRegisterFlagsNone);
    ASSERT_EQ(result, true);

    // Map the resource with error checking
    int count = 1;
    result = CudaCore::TryMapResource(count, &cudaResource);
    ASSERT_EQ(result, true);

    // Unmap the resource with error checking
    result = CudaCore::TryUnmapResource(count, &cudaResource);
    ASSERT_EQ(result, true);

    // Unregister the resource with error checking
    result = CudaCore::TryUnregisterResource(&cudaResource);
    ASSERT_EQ(result, true);

    EXPECT_EQ(cudaResource, nullptr);

    buffer->Release();
    device->Release();
    deviceContext->Release();
}

TEST(CudaCore, GetMappedPointer)
{
    // Declare device and device context pointers
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufferDesc.ByteWidth = sizeof(float) * 16;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    float initData[16] = { 0 };
    D3D11_SUBRESOURCE_DATA initDataDesc = {};
    initDataDesc.pSysMem = initData;

    ID3D11Buffer* buffer = nullptr;
    hr = device->CreateBuffer(&bufferDesc, &initDataDesc, &buffer);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA
    cudaGraphicsResource* cudaResource = nullptr;
    CudaCore::RegisterResource(&cudaResource, buffer, cudaGraphicsRegisterFlagsNone);

    // Map the resource
    int count = 1;
    CudaCore::MapResource(count, &cudaResource);

    // Get mapped pointer
    void* devPtr = nullptr;
    size_t size = (size_t)bufferDesc.ByteWidth;
    CudaCore::GetMappedPointer(&devPtr, &size, cudaResource);

    // Unmap the resource
    CudaCore::UnmapResource(count, &cudaResource);

    // Unregister the resource
    CudaCore::UnregisterResource(&cudaResource);

    buffer->Release();
    device->Release();
    deviceContext->Release();
}

TEST(CudaCore, GetMappedPointerTryVer)
{
    bool result = true;

    // Declare device and device context pointers with error checking
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation with error checking
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufferDesc.ByteWidth = sizeof(float) * 16;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    float initData[16] = { 0 };
    D3D11_SUBRESOURCE_DATA initDataDesc = {};
    initDataDesc.pSysMem = initData;

    ID3D11Buffer* buffer = nullptr;
    hr = device->CreateBuffer(&bufferDesc, &initDataDesc, &buffer);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA with error checking
    cudaGraphicsResource* cudaResource = nullptr;
    result = CudaCore::TryRegisterResource(&cudaResource, buffer, cudaGraphicsRegisterFlagsNone);
    ASSERT_EQ(result, true);

    // Map the resource with error checking
    int count = 1;
    result = CudaCore::TryMapResource(count, &cudaResource);
    ASSERT_EQ(result, true);

    // Get mapped pointer with error checking
    void* devPtr = nullptr;
    size_t size = (size_t)bufferDesc.ByteWidth;
    result = CudaCore::TryGetMappedPointer(&devPtr, &size, cudaResource);
    ASSERT_EQ(result, true);

    // Unmap the resource with error checking
    result = CudaCore::TryUnmapResource(count, &cudaResource);
    ASSERT_EQ(result, true);

    // Unregister the resource with error checking
    result = CudaCore::TryUnregisterResource(&cudaResource);
    ASSERT_EQ(result, true);

    buffer->Release();
    device->Release();
    deviceContext->Release();
}

TEST(CudaCore, GetMappedArray)
{
    // Declare device and device context pointers
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = 1024;
    texDesc.Height = 1024;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R32_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    ID3D11Texture2D* texture = nullptr;
    hr = device->CreateTexture2D(&texDesc, nullptr, &texture);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA
    cudaGraphicsResource* cudaResource = nullptr;
    CudaCore::RegisterResource(&cudaResource, texture, cudaGraphicsRegisterFlagsNone);

    // Map the resource
    int count = 1;
    CudaCore::MapResource(count, &cudaResource);

    // Get mapped array
    cudaArray_t array = nullptr;
    unsigned int arrayIndex = 0;
    unsigned int mipLevel = 0;
    CudaCore::GetMappedArray(&array, cudaResource, arrayIndex, mipLevel);

    // Unmap the resource
    CudaCore::UnmapResource(count, &cudaResource);

    // Unregister the resource
    CudaCore::UnregisterResource(&cudaResource);

    texture->Release();
    device->Release();
    deviceContext->Release();
}

TEST(CudaCore, GetMappedArrayTryVer)
{
    bool result = true;

    // Declare device and device context pointers with error checking
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;

    // Device Creation with error checking
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Use hardware driver
        nullptr,                    // Software rasterizer is not used
        0,                          // Flag is not set.
        nullptr,                    // No specific level of functionality required
        0,                          // Size of the above array
        D3D11_SDK_VERSION,          // SDK Version
        &device,
        &featureLevel,
        &deviceContext
    );

    if (FAILED(hr)) ASSERT_TRUE(false);

    D3D11_TEXTURE2D_DESC texDesc = {};
    texDesc.Width = 1024;
    texDesc.Height = 1024;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R32_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    ID3D11Texture2D* texture = nullptr;
    hr = device->CreateTexture2D(&texDesc, nullptr, &texture);
    if (FAILED(hr)) ASSERT_TRUE(false);

    // Register the resource with CUDA with error checking
    cudaGraphicsResource* cudaResource = nullptr;
    result = CudaCore::TryRegisterResource(&cudaResource, texture, cudaGraphicsRegisterFlagsNone);
    ASSERT_EQ(result, true);

    // Map the resource with error checking
    int count = 1;
    result = CudaCore::TryMapResource(count, &cudaResource);
    ASSERT_EQ(result, true);

    // Get mapped array with error checking
    cudaArray_t array = nullptr;
    unsigned int arrayIndex = 0;
    unsigned int mipLevel = 0;
    result = CudaCore::TryGetMappedArray(&array, cudaResource, arrayIndex, mipLevel);
    ASSERT_EQ(result, true);

    // Unmap the resource with error checking
    result = CudaCore::TryUnmapResource(count, &cudaResource);
    ASSERT_EQ(result, true);

    // Unregister the resource with error checking
    result = CudaCore::TryUnregisterResource(&cudaResource);
    ASSERT_EQ(result, true);

    texture->Release();
    device->Release();
    deviceContext->Release();
}