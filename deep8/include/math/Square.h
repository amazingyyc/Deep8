#ifndef DEEP8_MATH_SQUARE_H
#define DEEP8_MATH_SQUARE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * y = square(x)
 */
void Square(const Tensor &x, Tensor &y);

void SquareCPU(const Tensor &x, Tensor &y);

#ifdef HAVE_CUDA
void SquareGPU(const Tensor &x, Tensor &y);
#endif

/**
 * calculate the grad(x) of Square
 */
void SquareGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

void SquareGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);

#ifdef HAVE_CUDA
void SquareGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy);
#endif

}
}

#endif