#ifndef DEEP8_MATH_MATRIXMULTIPLY_H
#define DEEP8_MATH_MATRIXMULTIPLY_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**z = x * y*/
void MatrixMultiply(const Tensor &x, const Tensor &y, Tensor &z);

void MatrixMultiplyCPU(const Tensor &x, const Tensor &y, Tensor &z);

#ifdef HAVE_CUDA
void MatrixMultiplyGPU(const Tensor &x, const Tensor &y, Tensor &z);
#endif

/**gradient for x*/
void MatrixMultiplyGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

void MatrixMultiplyGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void MatrixMultiplyGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz);
#endif

/**gradient for y*/
void MatrixMultiplyGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

void MatrixMultiplyGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);

#ifdef HAVE_CUDA
void MatrixMultiplyGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz);
#endif

}
}

#endif