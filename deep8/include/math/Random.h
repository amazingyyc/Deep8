#ifndef DEEP8_MATH_RANDOM_H
#define DEEP8_MATH_RANDOM_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void Random(Tensor &x, float lower = 0, float upper = 1);

}
}

#endif