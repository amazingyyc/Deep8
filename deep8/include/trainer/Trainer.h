#ifndef DEEP8_TRAINER_H
#define DEEP8_TRAINER_H

#include "model/Tensor.h"
#include "model/Executor.h"
#include "nodes/Variable.h"
#include "trainer/LearningRateIterator.h"

namespace Deep8 {

class Trainer {
protected:
    int64_t steps;

    /**the learning rate*/
    LearningRateIterator *learningRate;

    /**for l2 penalty*/
    float weightDecay;

    explicit Trainer(LearningRateIterator *lr, float decay = 0);

public:
    virtual ~Trainer() = default;

    virtual void update(Executor *executor, Variable *parameter, float learningRate, float weightDecay, int64_t steps);

    void train(Executor *executor, std::vector<Variable*> parameters);
};



}

#endif //DEEP8_TRAINER_H
