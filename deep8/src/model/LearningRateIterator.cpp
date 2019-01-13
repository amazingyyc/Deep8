/**
 * Created by yanyuanchi on 2019/1/13.
 */

#include "LearningRateIterator.h"

namespace Deep8 {

/**constant learning rate*/
template <typename T>
ConstantLearningRateIterator<T>::ConstantLearningRateIterator(T lr): learningRate(lr) {
}


template <typename T>
T ConstantLearningRateIterator<T>::generateLearningRate(int64_t steps) {
    return this->learningRate;
}

DEEP8_DECLARATION_INSTANCE(ConstantLearningRateIterator)


/**linear decay*/
template <typename T>
LinearDecayLearningRateIterator<T>::LinearDecayLearningRateIterator(int64_t total, T start, T end)
        : totalSteps(total), startLearningRate(start), endLearningRate(end) {
}

template <typename T>
T LinearDecayLearningRateIterator<T>::generateLearningRate(int64_t steps) {
    return startLearningRate + (endLearningRate - startLearningRate) * T(steps) / T(totalSteps);
}

#ifdef HAVE_HALF
template <>
half LinearDecayLearningRateIterator<half>::generateLearningRate(int64_t steps) {
    float ratio = float(steps) / float(totalSteps);

    auto halfStart = __half2float(startLearningRate);
    auto halfEnd   = __half2float(endLearningRate);

    auto ret = halfStart + (halfEnd - halfStart) * float(steps) / float(totalSteps);

    return __float2half(ret);
}
#endif

DEEP8_DECLARATION_INSTANCE(LinearDecayLearningRateIterator)

}

