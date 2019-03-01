#include "math/AXPBY.h"
#include "trainer/SGDTrainer.h"

namespace Deep8 {

SGDTrainer::SGDTrainer(LearningRateIterator* lr, float decay) : Trainer(lr, decay) {
}

void SGDTrainer::update(Executor* executor, Variable* parameter, float learningRate, float weightDecay) {
    auto value    = parameter->value;
    auto gradient = parameter->gradient;

    float alpha = -1 * learningRate;
    float beta  = (1.0 - learningRate * weightDecay);

    Math::AXPBY(gradient, alpha, value, beta, value);
}


}