#ifndef DEEP8_MATH_RMSPROPUPDATE_H
#define DEEP8_MATH_RMSPROPUPDATE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**the RMSProp trainer update*/
void RMSPropUpdate(Tensor& value, Tensor& gradient, Tensor& accumulate, float rho, float epsilon, float learningRate, float weightDecay);

void RMSPropUpdateCPU(Tensor& value, Tensor& gradient, Tensor& accumulate, float rho, float epsilon, float learningRate, float weightDecay);

#ifdef HAVE_CUDA
void RMSPropUpdateGPU(Tensor& value, Tensor& gradient, Tensor& accumulate, float rho, float epsilon, float learningRate, float weightDecay);
#endif

}
}

#endif