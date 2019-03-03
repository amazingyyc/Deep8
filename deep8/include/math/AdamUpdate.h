#ifndef DEEP8_MATH_ADAMUPDATE_H
#define DEEP8_MATH_ADAMUPDATE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**the Adam trainer update*/
void AdamUpdate(Tensor& value, Tensor& gradient, Tensor& m, Tensor &v, float beta1, float beta2, float epsilon, float learningRate, float weightDecay, int64_t steps);

void AdamUpdateCPU(Tensor& value, Tensor& gradient, Tensor& m, Tensor& v, float beta1, float beta2, float epsilon, float learningRate, float weightDecay, int64_t steps);

#ifdef HAVE_CUDA
void AdamUpdateGPU(Tensor& value, Tensor& gradient, Tensor& m, Tensor& v, float beta1, float beta2, float epsilon, float learningRate, float weightDecay, int64_t steps);
#endif

}
}

#endif