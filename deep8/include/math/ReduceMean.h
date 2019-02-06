#ifndef DEEP8_MATH_REDUCEMEAN_H
#define DEEP8_MATH_REDUCEMEAN_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void ReduceMean(const Tensor &x, Tensor &y, int axis = -1);

void ReduceMeanCPU(const Tensor &x, Tensor &y, int axis = -1);

#ifdef HAVE_CUDA
void ReduceMeanGPU(const Tensor &x, Tensor &y, int axis = -1);
#endif

void ReduceMeanGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1);

void ReduceMeanGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1);

#ifdef HAVE_CUDA
void ReduceMeanGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1);
#endif

}
}