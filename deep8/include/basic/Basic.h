#ifndef DEEP8_BASIC_H
#define DEEP8_BASIC_H

#ifdef HAVE_CUDA
#define DEEP8_CUDA_FUNC __device__

#if _MSC_VER || __INTEL_COMPILER
#define DEEP8_CUDA_INLINE __forceinline
#else
#define DEEP8_CUDA_INLINE inline
#endif
#endif

#include <cstdint>
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
#include <typeindex>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <cfloat>

#ifdef __GUNC__
#include <mm_malloc.h>
#include <zconf.h>
#endif

#ifdef HAVE_HALF
#include <cuda_fp16.h>
#endif

#include <eigen/Eigen/Dense>
#include <eigen/unsupported/Eigen/CXX11/Tensor>
#include <eigen/unsupported/Eigen/CXX11/ThreadPool>

/**define the byte type*/
typedef unsigned char byte;

#define private public
#define protected public

#endif