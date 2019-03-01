#ifndef DEEP8_MATH_AXPBYPCZ_H
#define DEEP8_MATH_AXPBYPCZ_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * W = a * X + b * Y + c * Z
 * W can same with X or Y or Z
 */

void AXPBYPCZ(const Tensor& x, float a, const Tensor& y, float b, const Tensor& z, float c, Tensor &w);

void AXPBYPCZCPU(const Tensor& x, float a, const Tensor& y, float b, const Tensor& z, float c, Tensor& w);

#ifdef HAVE_CUDA
void AXPBYPCZGPU(const Tensor& x, float a, const Tensor& y, float b, const Tensor& z, float c, Tensor& w);
#endif

}
}

#endif