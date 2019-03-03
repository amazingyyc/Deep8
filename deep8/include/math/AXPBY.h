#ifndef DEEP8_MATH_AXPBY_H
#define DEEP8_MATH_AXPBY_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * Z = a * X + b * Y
 * Z can same with X or Y
 */

void AXPBY(const Tensor& x, float alpha, const Tensor& y, float beta, Tensor &z);

void AXPBYCPU(const Tensor& x, float alpha, const Tensor& y, float beta, Tensor& z);

#ifdef HAVE_CUDA
void AXPBYGPU(const Tensor& x, float alpha, const Tensor& y, float beta, Tensor& z);
#endif

}
}

#endif