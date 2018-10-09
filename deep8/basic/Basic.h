#ifndef DEEP8_BASIC_H
#define DEEP8_BASIC_H

#ifdef HAVE_CUDA

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
#define DEEP8_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#elif defined(__CUDACC_VER__)
#define DEEP8_CUDACC_VER __CUDACC_VER__
#else
#define DEEP8_CUDACC_VER 0
#endif

#if DEEP8_CUDACC_VER >= 70500
#define HAVE_HALF
#endif

#define DEEP8_CUDA_FUNC __device__

#if _MSC_VER || __INTEL_COMPILER
#define DEEP8_CUDA_INLINE __forceinline
#else
#define DEEP8_CUDA_INLINE inline
#endif
#endif

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cstddef>
#include <queue>
#include <random>
#include <utility>
#include <typeinfo>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#ifdef __GUNC__
#include <mm_malloc.h>
#include <zconf.h>
#endif

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

#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**define the byte type*/
typedef unsigned char byte;

#ifdef HAVE_HALF

#define DEEP8_DECLARATION_INSTANCE(name)    \
            template class name<float>;     \
            template class name<double>;    \
            template class name<half>;
#else
#define DEEP8_DECLARATION_INSTANCE(name)    \
            template class name<float>;     \
            template class name<double>;
#endif

#endif