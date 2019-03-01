#ifndef DEEP8_MATH_ADAGRADUPDATE_H
#define DEEP8_MATH_ADAGRADUPDATE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**the Adagrad trainer update*/
void AdagradUpdate(Tensor &value, Tensor &gradient, Tensor &accumulate, float epsilon, float learningRate, float weightDecay);

void AdagradUpdateCPU(Tensor& value, Tensor& gradient, Tensor& accumulate, float epsilon, float learningRate, float weightDecay);

#ifdef HAVE_CUDA
void AdagradUpdateGPU(Tensor& value, Tensor& gradient, Tensor& accumulate, float epsilon, float learningRate, float weightDecay);
#endif

}
}


#endif