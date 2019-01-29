#ifndef DEEP8_MATH_L1NORM_H
#define DEEP8_MATH_L1NORM_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {
/*
 * y = l1norm(x)
 */
void L1Norm(const Tensor &x, Tensor &y);

void L1NormCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void L1NormGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of L1Norm
 */
void L1NormGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void L1NormGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void L1NormGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif