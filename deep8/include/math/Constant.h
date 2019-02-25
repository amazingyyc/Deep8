#ifndef DEEP8_MATH_CONSTANT_H
#define DEEP8_MATH_CONSTANT_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void Constant(Tensor &x, float scalar);

void ConstantCPU(Tensor &x, float scalar);

#ifdef HAVE_CUDA
void ConstantGPU(Tensor &x, float scalar);
#endif

}
}

#endif