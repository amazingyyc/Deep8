#ifndef DEEP8_TYPE_H
#define DEEP8_TYPE_H

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
#if !defined(__CUDA_NO_HALF_OPERATORS__)
#if defined(__CUDACC__)

#define DEEP8_USE_HALF

#endif
#endif
#endif

#endif


#endif // !
