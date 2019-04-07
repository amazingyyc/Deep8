#include "trainer/Trainer.h"

namespace Deep8 {

Trainer::Trainer(LearningRateIterator *lr, float decay): learningRate(lr), weightDecay(decay), steps(0) {
}

void Trainer::update(Executor *executor, Variable *parameter, float learningRate, float weightDecay, int64_t steps) {
    DEEP8_RUNTIME_ERROR("can not call this function in trainer");
}

void Trainer::train(Executor *executor, std::vector<Variable*> parameters) {
    if (parameters.empty()) {
		return;
	}

    float lr = learningRate->next(steps);

    for (auto parameter : parameters) {
        if (!parameter->updateGradient) {
			continue;
		}

        this->update(executor, parameter, lr, this->weightDecay, steps);
    }

    steps++;
}

void Trainer::train(Executor *executor) {
    train(executor, executor->trainableParameters());
}




}