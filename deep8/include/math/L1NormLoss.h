#ifndef DEEP8_MATH_L1NORMLOSS_H
#define DEEP8_MATH_L1NORMLOSS_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/*
 * y = l1norm(x)
 */
void L1NormLoss(const Tensor &x, Tensor &y);

void L1NormLossCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void L1NormLossGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of L1Norm
 */
void L1NormLossGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void L1NormLossGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void L1NormLossGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif