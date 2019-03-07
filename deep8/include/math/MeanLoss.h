#ifndef DEEP8_MATH_MEANLOSS_H
#define DEEP8_MATH_MEANLOSS_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/*
 * y = mean(x)
 */
void MeanLoss(const Tensor& x, Tensor& y);

void MeanLossCPU(const Tensor& x, Tensor& y);

#ifdef HAVE_CUDA
void MeanLossGPU(const Tensor& x, Tensor& y);
#endif

/**
 * calculate the grad(x) of L1Norm
 */
void MeanLossGrad(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);

void MeanLossGradCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);

#ifdef HAVE_CUDA
void MeanLossGradGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy);
#endif

}
}

#endif