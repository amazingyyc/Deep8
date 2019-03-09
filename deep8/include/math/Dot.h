#ifndef DEEP8_MATH_DOT_H
#define DEEP8_MATH_DOT_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**z = dot(x, y)*/
void Dot(const Tensor &x, const Tensor &y, Tensor &z);

void DotCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void DotGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

void DotGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void DotGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void DotGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

void DotGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void DotGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void DotGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif
