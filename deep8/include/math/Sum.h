#ifndef DEEP8_MATH_SUM_H
#define DEEP8_MATH_SUM_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/*
 * y = sum(x)
 */
void Sum(const Tensor& x, Tensor& y);

void SumCPU(const Tensor& x, Tensor& y);

#ifdef HAVE_CUDA
void SumGPU(const Tensor& x, Tensor& y);
#endif

void SumGrad(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);

void SumGradCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);

#ifdef HAVE_CUDA
void SumGradGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);
#endif

}
}

#endif