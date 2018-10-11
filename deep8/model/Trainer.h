#ifndef DEEP8_TRAINER_H
#define DEEP8_TRAINER_H

#include "../basic/Tensor.h"
#include "../nodes/Parameter.h"

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

//#ifdef HAVE_CUDA
//
//template <typename real>
//__global__ void AdagradTrainerKernel(real *gradient, real scale, real *square, real *value, real epsilon, real learningRate, int N) {
//	int start  = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//	for (int i = start; i < N; i += stride) {
//		gradient[i] *= scale;
//		square[i] += gradient[i] * gradient[i];
//		value[i]  -= learningRate * gradient[i] / cuSqrt(square[i] + epsilon);
//	}
//}
//
//#endif
//
//template <typename T>
//class AdagradTrainer: public Trainer<T> {
//public:
//    T epsilon;
//    std::unordered_map<Parameter<T>*, Tensor<T>> accumulate;
//
//    explicit AdagradTrainer(T learningRate = 0.1, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0)
//            :Trainer<T>(learningRate, clipGradient, clipThreshold), epsilon(epsilon) {
//		check(epsilon);
//    }
//
//	template <typename real>
//	void check(real epsilon) {
//		DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
//	}
//
//#ifdef HAVE_HALF
//	template <>
//	void check<half>(half epsilon) {
//		DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
//	}
//#endif
//
//    ~AdagradTrainer() override {
//       accumulate.clear();
//    }
//
//protected: 
//	template <typename real>
//	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
//		auto value    = parameter->value;
//		auto gradient = parameter->gradient;
//
//		auto device = static_cast<CPUDevice*>(value.device());
//		auto eigenDevice = device->eigenDevice;
//
//		if (accumulate.find(parameter) == accumulate.end()) {
//			auto square = this->createTensorCPU(device, gradient.shape);
//			square.zero();
//
//			accumulate[parameter] = square;
//		}
//
//		auto square = accumulate[parameter];
//
//		eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
//		eTVec(square).device(*eigenDevice)  += eTVec(gradient).square();
//		eTVec(value).device(*eigenDevice)   -= eTVec(gradient) / (eTVec(square) + epsilon).sqrt() * this->learningRate;
//	}
//
//#ifdef HAVE_HALF
//	template <>
//	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
//		DEEP8_RUNTIME_ERROR("CPU not support half");
//	}
//#endif
//
//    void trainingCPU(Parameter<T> *parameter, T scale) override {
//		trainingCPUImpl(parameter, scale);
//    }
//
//#ifdef HAVE_CUDA
//	void trainingGPU(Parameter<T> *parameter, T scale) override {
//		auto value    = parameter->value;
//		auto gradient = parameter->gradient;
//
//		auto device = static_cast<GPUDevice*>(value.device());
//		auto size   = (int) gradient.size();
//
//		if (accumulate.find(parameter) == accumulate.end()) {
//			auto square = createTensorGPU(device, gradient.shape);
//			square.zero();
//
//			accumulate[parameter] = square;
//		}
//
//		auto square = accumulate[parameter];
//
//		int blockSize = 1024;
//		int grideSize = (size + blockSize - 1) / blockSize;
//
//		AdagradTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, square.data(), value.data(), epsilon, learningRate, size);
//	}
//#endif // HAVE_CUDA
//
//};
//
//#ifdef HAVE_CUDA
//
//template <typename real>
//__global__ void AdamTrainerKernel(real *gradient, real scale, real *mt, real *vt, real *value, real beta1, real beta2, real epsilon, real learningRate, int N) {
//	int start  = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//	for (int i = start; i < N; i += stride) {
//		gradient[i] *= scale;
//		mt[i] = mt[i] * beta1 + (real(1.0) - beta1) * gradient[i];
//		vt[i] = vt[i] * beta2 + gradient[i] * gradient[i] * (real(1.0) - beta2);
//		value[i] -= mt[i] / (cuSqrt(vt[i]) + epsilon) * learningRate;
//	}
//}
//
//#endif
//
//template <typename T>
//class AdamTrainer: public Trainer<T> {
//public:
//   T beta1;
//   T beta2;
//   T epsilon;
//
//   std::unordered_map<Parameter<T>*, Tensor<T>> m;
//   std::unordered_map<Parameter<T>*, Tensor<T>> v;
//
//   explicit AdamTrainer(T learningRate = 0.1, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0):
//		Trainer<T>(learningRate, clipGradient, clipThreshold), beta1(beta1), beta2(beta2), epsilon(epsilon) {
//	   check(epsilon);
//   }
//
//   template <typename real>
//   void check(real epsilon) {
//	   DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
//   }
//
//#ifdef HAVE_HALF
//   template <>
//   void check<half>(half epsilon) {
//	   DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
//   }
//#endif
//
//   ~AdamTrainer() override {
//       m.clear();
//       v.clear();
//   }
//
//protected:
//
//#ifdef HAVE_CUDA
//	template <typename real>
//	real calculateRealLearningRate(real learningRate, real beta1, real beta2, int64_t times) {
//		return learningRate * std::sqrt(1.0 - std::pow(beta2, real(times))) / (1 - std::pow(beta1, real(times)));
//	}
//
//#ifdef HAVE_HALF
//	template <>
//	half calculateRealLearningRate<half>(half learningRate, half beta1, half beta2, int64_t times) {
//		float learningRateF = __half2float(learningRate);
//		float beta1F        = __half2float(beta1);
//		float beta2F        = __half2float(beta2);
//
//		return __float2half(calculateRealLearningRate(learningRateF, beta1F, beta2F, times));
//	}
//#endif
//
//	void trainingGPU(Parameter<T> *parameter, T scale) override {
//		auto value    = parameter->value;
//		auto gradient = parameter->gradient;
//
//		auto device = static_cast<GPUDevice*>(value.device());
//
//		if (m.find(parameter) == m.end()) {
//			auto mt = createTensorGPU(device, gradient.shape);
//			mt.zero();
//
//			m[parameter] = mt;
//		}
//
//		if (v.find(parameter) == v.end()) {
//			auto vt = createTensorGPU(device, gradient.shape);
//			vt.zero();
//
//			v[parameter] = vt;
//		}
//
//		int size = (int) gradient.size();
//
//		auto mt = m[parameter];
//		auto vt = v[parameter];
//
//		int blockSize = 1024;
//		int grideSize = (size + blockSize - 1) / blockSize;
//
//		auto realLearningRate = calculateRealLearningRate(this->learningRate, beta1, beta2, this->times);
//
//		AdamTrainerKernel<T><<<grideSize, blockSize>>>(gradient.data(), scale, mt.data(), vt.data(), value.data(), beta1, beta2, epsilon, realLearningRate, size);
//	}
//
//#endif // HAVE_CUDA
//
//	template <typename real> 
//	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
//		auto value    = parameter->value;
//        auto gradient = parameter->gradient;
//
//		auto device = static_cast<CPUDevice*>(value.device());
//		auto eigenDevice = device->eigenDevice;
//
//		if (m.find(parameter) == m.end()) {
//			auto mt = createTensorCPU(device, gradient.shape);
//			mt.zero();
//
//			m[parameter] = mt;
//		}
//
//		if (v.find(parameter) == v.end()) {
//			auto vt = createTensorCPU(device, gradient.shape);
//			vt.zero();
//
//			v[parameter] = vt;
//		}
//
//		auto mt = m[parameter];
//		auto vt = v[parameter];
//
//		eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
//
//		eTVec(mt).device(*eigenDevice) = eTVec(mt) * beta1 + eTVec(gradient) * (1 - beta1);
//		eTVec(vt).device(*eigenDevice) = eTVec(vt) * beta2 + eTVec(gradient).square() * (1 - beta2);
//
//		auto realLearningRate = this->learningRate * sqrt(1 - std::pow(beta2, real(this->times))) / (1 - std::pow(beta1, real(this->times)));
//
//		eTVec(value).device(*eigenDevice) -= eTVec(mt) / (eTVec(vt).sqrt() + epsilon) * realLearningRate;
//	}
//
//#ifdef HAVE_HALF
//	template <>
//	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
//		DEEP8_RUNTIME_ERROR("CPU not support half");
//	}
//#endif
//
//    void trainingCPU(Parameter<T> *parameter, T scale) override {
//       trainingCPUImpl(parameter, scale);
//    }
//};
//
//#ifdef HAVE_CUDA
//template <typename real>
//__global__ void RMSPropTrainerKernel(real *gradient, real scale, real *vt, real *value, real decay, real epsilon, real learningRate, int N) {
//	int start = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//	for (int i = start; i < N; i += stride) {
//		gradient[i] *= scale;
//		vt[i] = vt[i] * decay + gradient[i] * gradient[i] * (real(1.0) - decay);
//		value[i] -= gradient[i] / cuSqrt(vt[i] + epsilon) * learningRate;
//	}
//}
//#endif
//
//template <typename T>
//class RMSPropTrainer: public Trainer<T> {
//public:
//   T decay;
//   T epsilon;
//
//   std::unordered_map<Parameter<T>*, Tensor<T>> v;
//
//   explicit RMSPropTrainer(T learningRate = 0.1, T decay = 0.9, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0):
//		Trainer<T>(learningRate, clipGradient, clipThreshold), decay(decay), epsilon(epsilon) {
//	   check(epsilon);
//   }
//
//   template <typename real>
//   void check(real epsilon) {
//	   DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
//   }
//
//#ifdef HAVE_HALF
//   template <>
//   void check<half>(half epsilon) {
//	   DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
//   }
//#endif
//
//   ~RMSPropTrainer() {
//       v.clear();
//   }
//
//protected:
//
//#ifdef HAVE_CUDA
//	void trainingGPU(Parameter<T> *parameter, T scale) override {
//		auto value    = parameter->value;
//		auto gradient = parameter->gradient;
//
//		auto device = static_cast<GPUDevice*>(value.device());
//		
//		if (v.find(parameter) == v.end()) {
//			auto vt = createTensorGPU(device, gradient.shape);
//			vt.zero();
//
//			v[parameter] = vt;
//		}
//
//		auto vt = v[parameter];
//
//		int size = (int)gradient.size();
//
//		int blockSize = 1024;
//		int grideSize = (size + blockSize - 1) / blockSize;
//
//		RMSPropTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, vt.data(), value.data(), decay, epsilon, learningRate, size);
//	}
//
//#endif // HAVE_CUDA
//
//	template <typename real>
//	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
//		auto value    = parameter->value;
//		auto gradient = parameter->gradient;
//
//		auto device = static_cast<CPUDevice*>(value.device());
//		auto eigenDevice = device->eigenDevice;
//
//		if (v.find(parameter) == v.end()) {
//			auto vt = createTensorCPU(device, gradient.shape);
//			vt.zero();
//
//			v[parameter] = vt;
//		}
//
//		auto vt = v[parameter];
//
//		eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
//		eTVec(vt).device(*eigenDevice)       = eTVec(vt) * decay + eTVec(gradient).square() * (1 - decay);
//		eTVec(value).device(*eigenDevice)   -= eTVec(gradient) / (eTVec(vt) + epsilon).sqrt() * this->learningRate;
//	}
//
//#ifdef HAVE_HALF
//	template <>
//	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
//		DEEP8_RUNTIME_ERROR("CPU not support half");
//	}
//#endif
//
//    void trainingCPU(Parameter<T> *parameter, T scale) override {
//	    trainingCPUImpl(parameter, scale);
//   }
//};
//
//#ifdef HAVE_CUDA
//
//template <typename real>
//__global__ void MomentumTrainerKernel(real *gradient, real scale, real *m, real *value, real alpha, real learningRate, int N) {
//	int start = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//	for (int i = start; i < N; i += stride) {
//		m[i] = m[i] * alpha - gradient[i] * learningRate * scale;
//		value[i] += m[i];
//	}
//}
//#endif // HAVE_CUDA
//
//template <typename T>
//class MomentumTrainer: public Trainer<T> {
//public:
//   T alpha;
//
//   std::unordered_map<Parameter<T>*, Tensor<T>> momentum;
//
//   explicit MomentumTrainer(T learningRate = 0.1, T alpha = 0.9, bool clipGradient = false, T clipThreshold = 5.0):
//		Trainer<T>(learningRate, clipGradient, clipThreshold), alpha(alpha) {
//   }
//
//   ~MomentumTrainer() {
//       momentum.clear();
//   }
//
//protected:
//
//#ifdef HAVE_CUDA
//
//	void trainingGPU(Parameter<T> *parameter, T scale) override {
//		auto value    = parameter->value;
//		auto gradient = parameter->gradient;
//
//		auto device = static_cast<GPUDevice*>(value.device());
//
//		if (momentum.find(parameter) == momentum.end()) {
//			auto m = createTensorGPU(device, gradient.shape);
//			m.zero();
//
//			momentum[parameter] = m;
//		}
//
//		auto m = momentum[parameter];
//
//		int size = (int)gradient.size();
//
//		int blockSize = 1024;
//		int grideSize = (size + blockSize - 1) / blockSize;
//
//		MomentumTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, m.data(), value.data(), alpha, learningRate, size);
//	}
//
//#endif // HAVE_CUDA
//	template <typename real>
//	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
//		auto value    = parameter->value;
//		auto gradient = parameter->gradient;
//
//		auto device = static_cast<CPUDevice*>(value.device());
//		auto eigenDevice = device->eigenDevice;
//
//		if (momentum.find(parameter) == momentum.end()) {
//			auto m = createTensorCPU(device, gradient.shape);
//			m.zero();
//
//			momentum[parameter] = m;
//		}
//
//		auto m = momentum[parameter];
//
//		eTVec(m).device(*eigenDevice) = eTVec(m) * alpha - eTVec(gradient) * this->learningRate * scale;
//		eTVec(value).device(*eigenDevice) += eTVec(m);
//	}
//
//#ifdef HAVE_HALF
//	template <>
//	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
//		DEEP8_RUNTIME_ERROR("CPU not support half");
//	}
//#endif
//
//   void trainingCPU(Parameter<T> *parameter, T scale) override {
//	   trainingCPUImpl(parameter, scale);
//   }
//};

}

#endif //DEEP8_TRAINER_H
