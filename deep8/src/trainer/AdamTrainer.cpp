#include "math/AdamUpdate.h"
#include "trainer/AdamTrainer.h"

namespace Deep8 {

AdamTrainer::AdamTrainer(LearningRateIterator* lr, float b1, float b2, float e, float decay): Trainer(lr, decay), beta1(b1), beta2(b2), epsilon(e) {
}

void AdamTrainer::update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) {
    auto value    = parameter->value;
    auto gradient = parameter->gradient;

    if (m.find(parameter) == m.end()) {
        auto &mt = executor->addVariable(value.shape, value.elementType, false);
        mt.value.zero();

        m[parameter] = &mt;
    }

    if (v.find(parameter) == v.end()) {
        auto &vt = executor->addVariable(value.shape, value.elementType, false);
        vt.value.zero();

        v[parameter] = &vt;
    }

    auto mt = m[parameter]->value;
    auto vt = v[parameter]->value;

    Math::AdamUpdate(value, gradient, mt, vt, beta1, beta2, this->epsilon, learningRate, weightDecay, steps);
}

}
