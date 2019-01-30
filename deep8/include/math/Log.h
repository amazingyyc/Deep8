#ifndef DEEP8_MATH_LOG_H
#define DEEP8_MATH_LOG_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = log(x)
 */
void Log(const Tensor &x, Tensor &y);

void LogCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void LogGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of log
 */
void LogGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void LogGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void LogGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif