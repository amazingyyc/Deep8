#ifndef DEEP8_MATH_POSITIVEUNITBALL_H
#define DEEP8_MATH_POSITIVEUNITBALL_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void positiveUnitball(Tensor &x);

void positiveUnitballCPU(Tensor &x);

#ifdef HAVE_CUDA
void positiveUnitballGPU(Tensor &x);
#endif

}
}

#endif