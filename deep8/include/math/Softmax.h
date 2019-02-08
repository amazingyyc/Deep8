#ifndef DEEP8_MATH_SOFTMAX_H
#define DEEP8_MATH_SOFTMAX_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void Softmax(const Tensor &x, Tensor &y, int axis = -1, void *ptr = nullptr);

void SoftmaxCPU(const Tensor &x, Tensor &y, int axis = -1, void *ptr = nullptr);

#ifdef HAVE_CUDA
void SoftmaxGPU(const Tensor &x, Tensor &y, int axis = -1, void *ptr = nullptr);
#endif

void SoftmaxGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1, void *ptr = nullptr);

void SoftmaxGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1, void *ptr = nullptr);

#ifdef HAVE_CUDA
void SoftmaxGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1, void *ptr = nullptr);
#endif

}
}