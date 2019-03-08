#ifndef DEEP8_MATH_MEAN_H
#define DEEP8_MATH_MEAN_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/*
 * y = mean(x)
 */
void Mean(const Tensor& x, Tensor& y);

void MeanCPU(const Tensor& x, Tensor& y);

#ifdef HAVE_CUDA
void MeanGPU(const Tensor& x, Tensor& y);
#endif

void MeanGrad(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);

void MeanGradCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);

#ifdef HAVE_CUDA
void MeanGradGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);
#endif

}
}

#endif