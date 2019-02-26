#ifndef DEEP8_MATH_UNIFORM_H
#define DEEP8_MATH_UNIFORM_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void Uniform(Tensor &x, float left = 0.0, float right = 1.0);

void UniformCPU(Tensor &x, float left = 0.0, float right = 1.0);

#ifdef HAVE_CUDA
void UniformGPU(Tensor &x, float left = 0.0, float right = 1.0);
#endif

}
}

#endif