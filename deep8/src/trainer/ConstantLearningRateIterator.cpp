#include "trainer/ConstantLearningRateIterator.h"

namespace Deep8 {

ConstantLearningRateIterator::ConstantLearningRateIterator(float lr): learningRate(lr) {
}

float ConstantLearningRateIterator::next(int64_t steps) {
    return learningRate;
}

}