#ifndef DEEP8_RMSPROPTRAINER_H
#define DEEP8_RMSPROPTRAINER_H

#include "model/Tensor.h"
#include "model/Executor.h"
#include "nodes/Variable.h"
#include "trainer/Trainer.h"

namespace Deep8 {

class RMSPropTrainer : public Trainer {
public:
    float rho;
    float epsilon;

    std::unordered_map<Variable*, Variable*> v;

    explicit RMSPropTrainer(LearningRateIterator* lr, float rho = 0.9, float epsilon = 1e-7, float decay = 0);

    void update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) override;
};

}

#endif