#ifndef DEEP8_TRAINER_H
#define DEEP8_TRAINER_H

#include "Tensor.h"
#include "Parameter.h"
#include "LearningRateIterator.h"
#include "Executor.h"

namespace Deep8 {

template <typename T>
class Trainer {
protected:
    /**clip the Gradient to void exploding gradient problem*/
    bool clipGradient;

    /**the clip threshold*/
    T clipThreshold;

    /**the learning rate iterator*/
    LearningRateIterator *learningRateIterator;

    explicit Trainer(LearningRateIterator *learningRate, bool clipGradient = false, T clipThreshold = 5.0);

protected:
	/**calculate the L2Norm of Parameter to void the exploding gradient problem*/
	T clipGradientScale(Executor *executor, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold);
	T clipGradientScaleCPU(Executor *executor,  Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold);

#ifdef HAVE_CUDA
	T clipGradientScaleGPU(Executor *executor, Device *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold);
#endif

	/**create a Tensor by shape*/
	Tensor<T> createTensorCPU(Device* device, Shape &shape);

#ifdef HAVE_CUDA
	Tensor<T> createTensorGPU(Device* device, Shape &shape);
#endif

    /**update the parameter*/
	virtual void updateCPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale);

#ifdef HAVE_CUDA
	virtual void updateGPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale);
#endif

public:
    virtual ~Trainer() = default;

    void update(Executor *executor, std::unordered_set<Parameter<T>*> &parameters, int64_t steps);
};

template <typename T>
class SGDTrainer: public Trainer<T> {
public:
    explicit SGDTrainer(LearningRateIterator *learningRate, bool clipGradient = false, T clipThreshold = 5.0);

protected:
	void updateCPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) override;

#ifdef HAVE_CUDA
	void updateGPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) override;
#endif
};

template <typename T>
class AdagradTrainer: public Trainer<T> {
public:
    T epsilon;
    // std::unordered_map<Parameter<T>*, Tensor<T>> accumulate;

    explicit AdagradTrainer(LearningRateIterator *learningRate, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0);

	void check(T epsilon);

protected:
	void updateCPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) override;

#ifdef HAVE_CUDA
	void updateGPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) override;
#endif
};

template <typename T>
class AdamTrainer: public Trainer<T> {
public:
   T beta1;
   T beta2;
   T epsilon;

   std::unordered_map<Parameter<T>*, Tensor<T>> m;
   std::unordered_map<Parameter<T>*, Tensor<T>> v;

   explicit AdamTrainer(T learningRate = 0.1, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0);

   void check(T epsilon);

   ~AdamTrainer() override;

protected:
	void trainingCPU(Parameter<T> *parameter, T scale) override;

#ifdef HAVE_CUDA
	T calculateRealLearningRate(T learningRate, T beta1, T beta2, int64_t times);
	void trainingGPU(Parameter<T> *parameter, T scale) override;
#endif
};

template <typename T>
class RMSPropTrainer: public Trainer<T> {
public:
   T decay;
   T epsilon;

   std::unordered_map<Parameter<T>*, Tensor<T>> v;

   explicit RMSPropTrainer(T learningRate = 0.1, T decay = 0.9, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0);

   void check(T epsilon);

   ~RMSPropTrainer();

protected:

	void trainingCPU(Parameter<T> *parameter, T scale) override;

#ifdef HAVE_CUDA
	void trainingGPU(Parameter<T> *parameter, T scale) override;
#endif
};

template <typename T>
class MomentumTrainer: public Trainer<T> {
public:
   T alpha;

   std::unordered_map<Parameter<T>*, Tensor<T>> momentum;

   explicit MomentumTrainer(T learningRate = 0.1, T alpha = 0.9, bool clipGradient = false, T clipThreshold = 5.0);

   ~MomentumTrainer();

protected:

	void trainingCPU(Parameter<T> *parameter, T scale) override;

#ifdef HAVE_CUDA
	void trainingGPU(Parameter<T> *parameter, T scale) override;
#endif
};

}

#endif //DEEP8_TRAINER_H
