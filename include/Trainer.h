#ifndef DEEP8_TRAINER_H
#define DEEP8_TRAINER_H

#include "Tensor.h"
#include "Parameter.h"

namespace Deep8 {

template <typename T>
class Trainer {
public:
	virtual ~Trainer() = default;

protected:
    /**the learning rate*/
    T learningRate;

    /**clip the Gradient to void exploding gradient problem*/
    bool clipGradient;

    /**the clip threshold*/
    T clipThreshold;

    /**the count of train*/
    int64_t times;

    explicit Trainer(T lr = 0.1, bool cg = false, T ct = 5.0):
            learningRate(lr), clipGradient(cg), clipThreshold(ct), times(0) {
    }

	T clipGradientScaleCPU(Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold);
	T clipGradientScaleGPU(Device *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold);

    /**
     * calculate the L2Norm of Parameter to void the exploding gradient problem
     */
	T clipGradientScale(std::unordered_set<Parameter<T>*> &parameters, T clipThreshold);


	Tensor<T> createTensorCPU(Device* device, Shape &shape);
	Tensor<T> createTensorGPU(Device* device, Shape &shape);

	/**the sub class implement the function*/
	virtual void trainingCPU(Parameter<T> *parameter, T scale) {};
	virtual void trainingGPU(Parameter<T> *parameter, T scale) {};

public:
	void training(std::unordered_set<Parameter<T>*> &parameters);
};

template <typename T>
class SGDTrainer: public Trainer<T> {
public:
    explicit SGDTrainer(T lr = 0.1, bool cg = false, T ct = 5.0): Trainer<T>(lr, cg, ct) {
    }

protected:
	void trainingCPU(Parameter<T> *parameter, T scale) override;
	void trainingGPU(Parameter<T> *parameter, T scale) override;
};

template <typename T>
class AdagradTrainer: public Trainer<T> {
public:
    T epsilon;
    std::unordered_map<Parameter<T>*, Tensor<T>> accumulate;

    explicit AdagradTrainer(T learningRate = 0.1, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0)
            :Trainer<T>(learningRate, clipGradient, clipThreshold), epsilon(epsilon) {
		check(epsilon);
    }

	template <typename real>
	void check(real epsilon) {
		DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
	}

#ifdef HAVE_HALF
	template <>
	void check<half>(half epsilon) {
		DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
	}
#endif

    ~AdagradTrainer() override {
       accumulate.clear();
    }

protected: 
	void trainingCPU(Parameter<T> *parameter, T scale) override;
	void trainingGPU(Parameter<T> *parameter, T scale) override;
};

template <typename T>
class AdamTrainer: public Trainer<T> {
public:
   T beta1;
   T beta2;
   T epsilon;

   std::unordered_map<Parameter<T>*, Tensor<T>> m;
   std::unordered_map<Parameter<T>*, Tensor<T>> v;

   explicit AdamTrainer(T learningRate = 0.1, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0):
		Trainer<T>(learningRate, clipGradient, clipThreshold), beta1(beta1), beta2(beta2), epsilon(epsilon) {
	   check(epsilon);
   }

   template <typename real>
   void check(real epsilon) {
	   DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
   }

#ifdef HAVE_HALF
   template <>
   void check<half>(half epsilon) {
	   DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
   }
#endif

   ~AdamTrainer() override {
       m.clear();
       v.clear();
   }

protected:
	T calculateRealLearningRate(T learningRate, T beta1, T beta2, int64_t times);

	void trainingGPU(Parameter<T> *parameter, T scale) override;
	void trainingCPU(Parameter<T> *parameter, T scale) override;
};



template <typename T>
class RMSPropTrainer: public Trainer<T> {
public:
   T decay;
   T epsilon;

   std::unordered_map<Parameter<T>*, Tensor<T>> v;

   explicit RMSPropTrainer(T learningRate = 0.1, T decay = 0.9, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0):
		Trainer<T>(learningRate, clipGradient, clipThreshold), decay(decay), epsilon(epsilon) {
	   check(epsilon);
   }

   template <typename real>
   void check(real epsilon) {
	   DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
   }

#ifdef HAVE_HALF
   template <>
   void check<half>(half epsilon) {
	   DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
   }
#endif

   ~RMSPropTrainer() {
       v.clear();
   }

protected:

	void trainingGPU(Parameter<T> *parameter, T scale) override;
	void trainingCPU(Parameter<T> *parameter, T scale) override;
};



template <typename T>
class MomentumTrainer: public Trainer<T> {
public:
   T alpha;

   std::unordered_map<Parameter<T>*, Tensor<T>> momentum;

   explicit MomentumTrainer(T learningRate = 0.1, T alpha = 0.9, bool clipGradient = false, T clipThreshold = 5.0):
		Trainer<T>(learningRate, clipGradient, clipThreshold), alpha(alpha) {
   }

   ~MomentumTrainer() {
       momentum.clear();
   }

protected:

	void trainingGPU(Parameter<T> *parameter, T scale) override;
	void trainingCPU(Parameter<T> *parameter, T scale) override;
};

}

#endif //DEEP8_TRAINER_H
