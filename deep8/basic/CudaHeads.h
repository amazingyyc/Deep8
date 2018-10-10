#ifndef DEEP8_CUDAHEADS_H
#define DEEP8_CUDAHEADS_H

#include "Basic.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
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

#endif

#endif