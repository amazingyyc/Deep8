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

    explicit Trainer(LearningRateIterator *lr, float deacy = 0);

public:
    virtual ~Trainer() = default;

    virtual void update(Executor *executor, Variable *parameter, float learningRate, float weightDecay, int64_t steps);

    void update(Executor *executor, std::unordered_set<Variable*> &parameters);
};




// template <typename T>
// class AdamTrainer: public Trainer<T> {
// public:
//    T beta1;
//    T beta2;
//    T epsilon;

//    std::unordered_map<Parameter<T>*, Tensor<T>> m;
//    std::unordered_map<Parameter<T>*, Tensor<T>> v;

//    explicit AdamTrainer(T learningRate = 0.1, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0);

//    void check(T epsilon);

//    ~AdamTrainer() override;

// protected:
// 	void trainingCPU(Parameter<T> *parameter, T scale) override;

// #ifdef HAVE_CUDA
// 	T calculateRealLearningRate(T learningRate, T beta1, T beta2, int64_t times);
// 	void trainingGPU(Parameter<T> *parameter, T scale) override;
// #endif
// };

// template <typename T>
// class RMSPropTrainer: public Trainer<T> {
// public:
//    T decay;
//    T epsilon;

//    std::unordered_map<Parameter<T>*, Tensor<T>> v;

//    explicit RMSPropTrainer(T learningRate = 0.1, T decay = 0.9, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0);

//    void check(T epsilon);

//    ~RMSPropTrainer();

// protected:

// 	void trainingCPU(Parameter<T> *parameter, T scale) override;

// #ifdef HAVE_CUDA
// 	void trainingGPU(Parameter<T> *parameter, T scale) override;
// #endif
// };



}

#endif //DEEP8_TRAINER_H
