#ifndef DEEP8_MATH_REDUCEMEAN_H
#define DEEP8_MATH_REDUCEMEAN_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * axis the dimension that to be reduce
 * keepDims if keep the y output dimension
 * includeBatch in the reduce include the batch
 * like x shape is (batch, [dim0, dim1, dim2])
 * if includeBatch is true than the rank is 4, so axis can be in [-4, 3)
 * if includeBatch is false than the rank is 3, than the axis must be in [-3, 2)
 */
void ReduceMean(const Tensor &x, Tensor &y, std::vector<int> &axis, bool keepDims, bool includeBatch);

void ReduceMeanCPU(const Tensor &x, Tensor &y, std::vector<int> &axis, bool keepDims, bool includeBatch);

#ifdef HAVE_CUDA
void ReduceMeanGPU(const Tensor &x, Tensor &y, std::vector<int> &axis, bool keepDims, bool includeBatch);
#endif

void ReduceMeanGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, std::vector<int> &axis, bool keepDims, bool includeBatch);

void ReduceMeanGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, std::vector<int> &axis, bool keepDims, bool includeBatch);

#ifdef HAVE_CUDA
void ReduceMeanGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, std::vector<int> &axis, bool keepDims, bool includeBatch);
#endif

}
}

#endif