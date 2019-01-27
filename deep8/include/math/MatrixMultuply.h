#ifndef DEEP8_MATH_MATRIXMULTIPLY_H
#define DEEP8_MATH_MATRIXMULTIPLY_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void MatrixMultiply(const Tensor &x, const Tensor &y, Tensor &z);

void MatrixMultiplyCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void MatrixMultiplyGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

}
}

#endif