#ifndef DEEP8_MATH_CROSSENTROPY_H
#define DEEP8_MATH_CROSSENTROPY_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void CrossEntropy(const Tensor &x, const Tensor &y, Tensor &z);

void CrossEntropyCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void CrossEntropyGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

void CrossEntropyGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void CrossEntropyGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void CrossEntropyGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

void CrossEntropyGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void CrossEntropyGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void CrossEntropyGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif