#ifndef DEEP8_MATH_LOGSOFTMAX_H
#define DEEP8_MATH_LOGSOFTMAX_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void LogSoftmax(const Tensor &x, Tensor &y, int axis = -1, void *maxptr = nullptr, void *sumptr = nullptr);

void LogSoftmaxCPU(const Tensor &x, Tensor &y, int axis = -1, void *maxptr = nullptr, void *sumptr = nullptr);

#ifdef HAVE_CUDA
void LogSoftmaxGPU(const Tensor &x, Tensor &y, int axis = -1, void *maxptr = nullptr, void *sumptr = nullptr);
#endif

void LogSoftmaxGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1, void *sumptr = nullptr);

void LogSoftmaxGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1, void *sumptr = nullptr);

#ifdef HAVE_CUDA
void LogSoftmaxGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis = -1, void *sumptr = nullptr);
#endif

    
}
}


#endif