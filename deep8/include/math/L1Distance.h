#ifndef DEEP8_MATH_L1DISTANCE_H
#define DEEP8_MATH_L1DISTANCE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**z = l1_distance(x, y)*/
void L1Distance(const Tensor &x, const Tensor &y, Tensor &z);

void L1DistanceCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void L1DistanceGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

void L1DistanceGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void L1DistanceGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void L1DistanceGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

void L1DistanceGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void L1DistanceGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void L1DistanceGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif
