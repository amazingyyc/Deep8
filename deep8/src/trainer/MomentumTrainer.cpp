#include "math/AXPBYPCZ.h"
#include "math/AXPBY.h"
#include "trainer/MomentumTrainer.h"

namespace Deep8 {

MomentumTrainer::MomentumTrainer(LearningRateIterator* lr, float a, float decay) : Trainer(lr, decay), alpha(a) {
}

void MomentumTrainer::update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) {
    auto value    = parameter->value;
    auto gradient = parameter->gradient;

    if (momentums.find(parameter) == momentums.end()) {
        auto mv = executor->addVariable(value.shape, value.elementType, false);
        mv->value.zero();

        momentums[parameter] = mv;
    }

    auto mv = momentums[parameter]->value;

    float a = -1 * learningRate;
    float b = -1 * learningRate * weightDecay;
    float c = this->alpha;

    Math::AXPBYPCZ(gradient, a, value, b, mv, c, mv);
    Math::AXPBY(mv, 1.0, value, 1.0, value);
}


}