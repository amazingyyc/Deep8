#ifndef DEEP8_MATH_REDUCESUM_H
#define DEEP8_MATH_REDUCESUM_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * axis the dimension that to be reduce
 * keepDims if keep the y output dimension
 * like x shape is (batch, [dim0, dim1, dim2])
 * than the rank is 4, so axis can be in [-4, 3)
 */
void ReduceSum(const Tensor& x, Tensor& y, const std::vector<int>& axis, bool keepDims);

void ReduceSumCPU(const Tensor& x, Tensor& y, const std::vector<int>& axis, bool keepDims);

#ifdef HAVE_CUDA
void ReduceSumGPU(const Tensor& x, Tensor& y, const std::vector<int>& axis, bool keepDims);
#endif

void ReduceSumGrad(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy, const std::vector<int>& axis, bool keepDims);

void ReduceSumGradCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy, const std::vector<int>& axis, bool keepDims);

#ifdef HAVE_CUDA
void ReduceSumGradGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy, const std::vector<int>& axis, bool keepDims);
#endif

}
}

#endif