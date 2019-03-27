#ifndef DEEP8_MATH_PRELU_H
#define DEEP8_MATH_PRELU_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * z = prelu(x, y)
 */
void PReLu(const Tensor& x, const Tensor& y, Tensor &z);

void PReLuCPU(const Tensor& x, const Tensor& y, Tensor& z);

#ifdef HAVE_CUDA
void PReLuGPU(const Tensor& x, const Tensor& y, Tensor& z);
#endif

void PReLuGradX(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz);

void PReLuGradXCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz);

#ifdef HAVE_CUDA
void PReLuGradXGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz);
#endif

void PReLuGradY(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz);

void PReLuGradYCPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz);

#ifdef HAVE_CUDA
void PReLuGradYGPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz);
#endif

}
}

#endif