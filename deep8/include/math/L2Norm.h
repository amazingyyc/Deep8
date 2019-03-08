#ifndef DEEP8_MATH_L2NORM_H
#define DEEP8_MATH_L2NORM_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/*
 * y = l2norm(x)
 */
void L2Norm(const Tensor &x, Tensor &y);

void L2NormCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void L2NormGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of L1Norm
 */
void L2NormGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void L2NormGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void L2NormGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif
