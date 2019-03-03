#include "nodes/Parameter.h"

namespace Deep8 {

Parameter::Parameter(): Variable() {
}

Parameter::Parameter(Tensor &value): Variable(value) {
}

Parameter::Parameter(Tensor &value, Tensor &gradient): Variable(value, gradient) {
}


}