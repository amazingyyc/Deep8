#ifndef DEEP8_DEEP8_H
#define DEEP8_DEEP8_H

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
#define EIGEN_HAS_CUDA_FP16

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "basic/Exception.h"
#include "basic/CudaException.h"
#include "basic/MemoryAllocator.h"
#include "basic/Shape.h"

#include "utils/Utils.h"
#include "utils/MathUtils.h"
#include "utils/CudaMathUtils.h"

#include "basic/CPUMemoryPool.h"
#include "basic/Device.h"
#include "basic/Tensor.h"

#include "utils/ShapeUtils.h"
#include "utils/TensorUtils.h"

#include "nodes/Node.h"
#include "nodes/Variable.h"
#include "nodes/Parameter.h"
#include "nodes/InputParameter.h"
#include "nodes/ConstantParameter.h"
#include "nodes/Function.h"

#include "nodes/Abs.h"
#include "nodes/Add.h"
#include "nodes/AddScalar.h"
#include "nodes/AvgPooling2d.h"
#include "nodes/Conv2d.h"
#include "nodes/DeConv2d.h"
#include "nodes/Divide.h"
#include "nodes/DivideScalar.h"
#include "nodes/Exp.h"
#include "nodes/L1Norm.h"
#include "nodes/L2Norm.h"
#include "nodes/Linear.h"
#include "nodes/Log.h"
#include "nodes/LReLu.h"
#include "nodes/MatrixMultiply.h"
#include "nodes/MaxPooling2d.h"
#include "nodes/Minus.h"
#include "nodes/MinusScalar.h"
#include "nodes/Multiply.h"
#include "nodes/MultiplyScalar.h"
#include "nodes/Pow.h"
#include "nodes/ReLu.h"
#include "nodes/ReShape.h"
#include "nodes/ScalarDivide.h"
#include "nodes/ScalarMinus.h"
#include "nodes/Sigmoid.h"
#include "nodes/Softmax.h"
#include "nodes/Square.h"
#include "nodes/SumElements.h"
#include "nodes/TanH.h"

#include "model/TensorInit.h"
#include "model/Trainer.h"
#include "model/Executor.h"
#include "model/Expression.h"
#include "model/DefaultExecutor.h"
#include "model/PreDefinition.h"

#endif