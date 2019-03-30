#include "math/AdagradUpdate.h"
#include "trainer/AdagradTrainer.h"

namespace Deep8 {

AdagradTrainer::AdagradTrainer(LearningRateIterator* lr, float e, float decay): Trainer(lr, decay), epsilon(e) {
}

void AdagradTrainer::update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) {
    auto value    = parameter->value;
    auto gradient = parameter->gradient;

    if (accumulates.find(parameter) == accumulates.end()) {
        auto accu = executor->addVariable(value.shape, value.elementType, false);
        accu.value.zero();

        accumulates[parameter] = &accu;
    }
    
    auto accu = accumulates[parameter]->value;

    Math::AdagradUpdate(value, gradient, accu, this->epsilon, learningRate, weightDecay);
}



}