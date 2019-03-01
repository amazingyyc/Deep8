#ifndef DEEP8_MOMENTUMTRAINER_H
#define DEEP8_MOMENTUMTRAINER_H

#include "model/Tensor.h"
#include "model/Executor.h"
#include "nodes/Variable.h"
#include "trainer/Trainer.h"


namespace Deep8 {

class MomentumTrainer : public Trainer {
public:
    float alpha;

    std::unordered_map<Variable*, Variable*> momentums;

    explicit MomentumTrainer(LearningRateIterator* lr, float alpha = 0.9, float deacy = 0);

    void update(Executor* executor, Variable* parameter, float learningRate, float weightDecay) override;

};

}