#include "trainer/Trainer.h"

namespace Deep8 {

Trainer::Trainer(LearningRateIterator *lr, float deacy): learningRate(lr), weightDecay(deacy), steps(0) {
}

void Trainer::update(Executor *executor, Variable *parameter, float learningRate, float weightDecay, int64_t steps) {
    DEEP8_RUNTIME_ERROR("can not call this function in trainer");
}

void Trainer::update(Executor *executor, std::unordered_set<Variable*> &parameters) {
    if (parameters.empty()) {
		return;
	}

    float lr = learningRate->nextLearningRate(steps);

    for (auto parameter : parameters) {
        if (!parameter->updateGradient) {
			continue;
		}

        this->update(executor, parameter, lr, this->weightDecay, steps);
    }

    steps++;
}





}