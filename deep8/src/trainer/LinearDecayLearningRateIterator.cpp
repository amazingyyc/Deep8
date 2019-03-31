#include "trainer/LinearDecayLearningRateIterator.h"

namespace Deep8 {

LinearDecayLearningRateIterator::LinearDecayLearningRateIterator(int64_t total, float start, float end)
    :totalSteps(total), startLearningRate(start), endLearningRate(end) {
}

float LinearDecayLearningRateIterator::next(int64_t steps) {
    return (float(steps) / float(totalSteps)) * (endLearningRate - startLearningRate) + startLearningRate;
}


}