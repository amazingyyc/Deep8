#include "math/Uniform.h"
#include "math/Random.h"

namespace Deep8 {
namespace Math {

void Random(Tensor &x, float lower, float upper) {
    Uniform(x, lower, upper);
}

}
}