#ifndef DEEP8_MATH_GAUSSIAN_H
#define DEEP8_MATH_GAUSSIAN_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void Gaussian(Tensor &x, float mean = 0.0, float stddev = 0.1);

void GaussianCPU(Tensor &x, float mean = 0.0, float stddev = 0.1);

#ifdef HAVE_CUDA
void GaussianGPU(Tensor &x, float mean = 0.0, float stddev = 0.1);
#endif

}
}

#endif