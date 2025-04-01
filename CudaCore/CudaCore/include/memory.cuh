#pragma once

#include "CudaCore/include/config.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CudaCore
{

CUDA_CORE void CheckCuda(cudaError_t call);

CUDA_CORE void SampleFunc(int& num);

}