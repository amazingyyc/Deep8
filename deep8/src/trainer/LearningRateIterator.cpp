#include "trainer/LearningRateIterator.h"

namespace Deep8 {

float LearningRateIterator::next(int64_t steps) {
    DEEP8_RUNTIME_ERROR("cann not call function nextLearningRate in LearningRateIterator");
}

}