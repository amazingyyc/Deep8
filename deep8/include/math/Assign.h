#ifndef DEEP8_MATH_ASSIGN_H
#define DEEP8_MATH_ASSIGN_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void Assign(const Tensor &x, Tensor &y);

void AssignCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void AssignGPU(const Tensor &x, Tensor &y);
#endif

}
}

#endif
