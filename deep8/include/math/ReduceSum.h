#ifndef DEEP8_MATH_REDUCESUM_H
#define DEEP8_MATH_REDUCESUM_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void ReduceSum(const Tensor &x, Tensor &y, int axis = -1);

void ReduceSumCPU(const Tensor &x, Tensor &y, int axis = -1);

#ifdef HAVE_CUDA
void ReduceSumGPU(const Tensor &x, Tensor &y, int axis = -1);
#endif

void ReduceSumGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1);

void ReduceSumGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1);

#ifdef HAVE_CUDA
void ReduceSumGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1);
#endif

}
}

#endif