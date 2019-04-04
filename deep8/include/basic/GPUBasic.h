#ifndef DEEP8_GPUBASIC_H
#define DEEP8_GPUBASIC_H

#include "basic/Basic.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <cublas_v2.h>
#include <curand.h>
#include <math_functions.h>
#include <cuda_occupancy.h>

#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif

#ifdef HAVE_HALF
#include <cuda_fp16.h>
#endif

#define DEEP8_GPU_BLOCK_SIZE 1024

#endif

#endif