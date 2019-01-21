#ifndef DEEP8_MATH_ADD_H
#define DEEP8_MATH_ADD_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * z = x + y
 * support broadcast
 */
void Add(const Tensor &x, const Tensor &y, Tensor &z);

void AddCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void AddGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

}
}

#endif