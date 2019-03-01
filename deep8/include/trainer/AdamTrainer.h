#ifndef DEEP8_ADAMTRAINER_H
#define DEEP8_ADAMTRAINER_H

#include "model/Tensor.h"
#include "model/Executor.h"
#include "nodes/Variable.h"
#include "trainer/Trainer.h"

namespace Deep8 {

class AdamTrainer : public Trainer {
public:
    float beta1;
    float beta2;
    float epsilon;

    std::unordered_map<Variable*, Variable*> m;
    std::unordered_map<Variable*, Variable*> v;

    explicit AdamTrainer(LearningRateIterator* lr, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 0.9, float decay = 0);

    void update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) override;

};


}

#endif