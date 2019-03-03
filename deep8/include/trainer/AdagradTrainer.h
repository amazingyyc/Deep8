#ifndef DEEP8_ADAGRADTRAINER_H
#define DEEP8_ADAGRADTRAINER_H

#include "model/Tensor.h"
#include "model/Executor.h"
#include "nodes/Variable.h"
#include "trainer/Trainer.h"

namespace Deep8 {

class AdagradTrainer : public Trainer {
public:
    float epsilon;

    std::unordered_map<Variable*, Variable*> accumulates;

    explicit AdagradTrainer(LearningRateIterator* lr, float epsilon = 1e-7, float decay = 0);

    void update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) override;

};

}

#endif