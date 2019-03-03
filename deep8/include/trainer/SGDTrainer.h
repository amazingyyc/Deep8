#ifndef DEEP8_SGDTRAINER_H
#define DEEP8_SGDTRAINER_H

#include "model/Tensor.h"
#include "model/Executor.h"
#include "nodes/Variable.h"
#include "trainer/Trainer.h"

namespace Deep8 {

class SGDTrainer : public Trainer {
public:
    explicit SGDTrainer(LearningRateIterator* lr, float decay = 0);

    void update(Executor* executor, Variable* parameter, float learningRate, float weightDecay, int64_t steps) override;
};

}

#endif