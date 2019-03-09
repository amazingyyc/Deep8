#ifndef DEEP8_MATH_L2DISTANCE_H
#define DEEP8_MATH_L2DISTANCE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**z = l2_distance(x, y)*/
void L2Distance(const Tensor &x, const Tensor &y, Tensor &z);

void L2DistanceCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void L2DistanceGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

void L2DistanceGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void L2DistanceGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void L2DistanceGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

void L2DistanceGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void L2DistanceGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void L2DistanceGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif
