#include "math/RMSPropUpdate.h"
#include "trainer/RMSPropTrainer.h"

namespace Deep8 {

RMSPropTrainer::RMSPropTrainer(LearningRateIterator* lr, float r, float e, float decay): Trainer(lr, decay), rho(r), epsilon(e){
}

void RMSPropTrainer::update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) {
    auto value = parameter->value;
    auto gradient = parameter->gradient;

    if (v.find(parameter) == v.end()) {
        auto &vt = executor->addVariable(value.shape, value.elementType, false);
        vt.value.zero();

        v[parameter] = &vt;
    }

    auto vt = v[parameter]->value;

    Math::RMSPropUpdate(value, gradient, vt, this->rho, this->epsilon, learningRate, weightDecay);
}

}