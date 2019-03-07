#ifndef DEEP8_MATH_L2NORMLOSS_H
#define DEEP8_MATH_L2NORMLOSS_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {
/*
 * y = l1norm(x)
 */
void L2NormLoss(const Tensor &x, Tensor &y);

void L2NormLossCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void L2NormLossGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of L1Norm
 */
void L2NormLossGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void L2NormLossGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void L2NormLossGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif