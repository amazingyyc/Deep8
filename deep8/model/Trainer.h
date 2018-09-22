#ifndef DEEP8_TRAINER_H
#define DEEP8_TRAINER_H

namespace Deep8 {

enum class TrainerType {
    SGD,
    Adagrad,
    Adam,
    RMSProp,
    Momentum,
};

#ifdef HAVE_CUDA
#ifdef HAVE_HALF

template <int blockSize>
__global__ void TrainerNorm2HalfKernel(const half *x, float *y, const int size) {
	SharedMemory<float> shareMemory;
	float *shared = shareMemory.pointer();

	int threaId = threadIdx.x;

	int j = threaId;

	shared[threaId] = 0;

	while (j < size) {
		shared[threaId] += __half2float(x[j]) * __half2float(x[j]);

		j += blockSize;
	}

	__syncthreads();

	if (blockSize >= 1024) {
		if (threaId < 512) {
			shared[threaId] += shared[threaId + 512];
		}

		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threaId < 256) {
			shared[threaId] += shared[threaId + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threaId < 128) {
			shared[threaId] += shared[threaId + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threaId < 64) {
			shared[threaId] += shared[threaId + 64];
		}

		__syncthreads();
	}

	if (threaId < 32) {
		warpSumReduce<blockSize, float>(shared, threaId);
	}
		
	if (0 == threaId) {
		y[0] = shared[threaId];
	}
}

#endif // HAVE_HALF
#endif

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

	template <typename real>
	real clipGradientScaleCPUImpl(Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<real>*> &parameters, real clipThreshold) {
		std::vector<real> l2NormVec;

		for (auto node : parameters) {
			if (!node->updateGradient) {
				continue;
			}

			auto parameter = node;
			auto gradient = parameter->gradient;

			l2NormVec.push_back(real(0));

			Eigen::TensorMap<Eigen::Tensor<real, 0, Eigen::RowMajor>> sum(static_cast<real*>(&(l2NormVec[l2NormVec.size() - 1])));
			Eigen::TensorMap<Eigen::Tensor<real, 1, Eigen::RowMajor>> vec(gradient.data(), gradient.size());

			sum.device(*device) = vec.square().sum();
		}

		real sum = 0;

		for (auto item : l2NormVec) {
			sum += item;
		}

		auto scale = clipThreshold / std::sqrt(sum);

		if (isnan(scale) || isinf(scale)) {
			return real(1);
		}

		return scale;
	}

#ifdef HAVE_HALF
	template <>
	half clipGradientScaleCPUImpl<half>(Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<half>*> &parameters, half clipThreshold) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

    T clipGradientScaleCPU(Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
		return clipGradientScaleCPUImpl(device, parameters, clipThreshold);
    }

#ifdef HAVE_CUDA
	float clipGradientScaleGPU(GPUDevice *device, std::unordered_set<Parameter<float>*> &parameters, float clipThreshold) {
		std::vector<float> l2NormVec;

		for (auto node : parameters) {
			if (!node->updateGradient) {
				continue;
			}

			auto parameter = node;
			auto gradient  = parameter->gradient;

			l2NormVec.push_back(float(0));

			CUBLAS_CHECK(cublasSnrm2(device->cublasHandle, (int)gradient.size(), gradient.data(), 1, &(l2NormVec[l2NormVec.size() - 1])));
		}

		float sum = 0;

		for (auto item : l2NormVec) {
			sum += item;
		}


		auto scale = clipThreshold / std::sqrt(sum);

		if (isnan(scale) || isinf(scale)) {
			return 1;
		}

		return scale;
	}

	double clipGradientScaleGPU(GPUDevice *device, std::unordered_set<Parameter<double>*> &parameters, double clipThreshold) {
		std::vector<double> l2NormVec;

		for (auto node : parameters) {
			if (!node->updateGradient) {
				continue;
			}

			auto parameter = node;
			auto gradient  = parameter->gradient;

			l2NormVec.push_back(double(0));

			CUBLAS_CHECK(cublasDnrm2(device->cublasHandle, (int)gradient.size(), gradient.data(), 1, &(l2NormVec[l2NormVec.size() - 1])));
		}

		double sum = 0;

		for (auto item : l2NormVec) {
			sum += item;
		}

		auto scale = clipThreshold / std::sqrt(sum);

		if (isnan(scale) || isinf(scale)) {
			return 1;
		}

		return scale;
	}

#ifdef HAVE_HALF
	half clipGradientScaleGPU(GPUDevice *device, std::unordered_set<Parameter<half>*> &parameters, half clipThreshold) {
		int updateCount = 0;

		for (auto node : parameters) {
			if (node->updateGradient) {
				updateCount++;
			}
		}

		if (0 >= updateCount) {
			return 1.0;
		}

		float *sumPtr = (float*)device->malloc(sizeof(float) * updateCount);

		int index = 0;

		for (auto node : parameters) {
			if (!node->updateGradient) {
				continue;
			}

			auto parameter = node;
			auto gradient  = parameter->gradient;
			
			int size = (int)gradient.size();

			int blockSize = 1024;

			if (size < blockSize) {
				blockSize = prevPowerOf2(size);
			}

			int sharedSize = sizeof(float) * blockSize;

			if (1024 == blockSize) {
				TrainerNorm2HalfKernel<1024> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (512 == blockSize) {
				TrainerNorm2HalfKernel<512> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (256 == blockSize) {
				TrainerNorm2HalfKernel<256> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (128 == blockSize) {
				TrainerNorm2HalfKernel<128> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (64 == blockSize) {
				TrainerNorm2HalfKernel<64> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (32 == blockSize) {
				TrainerNorm2HalfKernel<32> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (16 == blockSize) {
				TrainerNorm2HalfKernel<16> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (8 == blockSize) {
				TrainerNorm2HalfKernel<8> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (4 == blockSize) {
				TrainerNorm2HalfKernel<4> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (2 == blockSize) {
				TrainerNorm2HalfKernel<2> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else if (1 == blockSize) {
				TrainerNorm2HalfKernel<1> << <1, blockSize, sharedSize >> > (gradient.data(), sumPtr + index, size);
			} else {
				DEEP8_RUNTIME_ERROR("the block size is error");
			}

			index++;
		}

		std::vector<float> l2NormVec(updateCount);

		device->copyFromGPUToCPU(sumPtr, &l2NormVec[0], sizeof(float) * updateCount);
		device->free(sumPtr);

		float sum = 0;

		for (auto item : l2NormVec) {
			sum += item;
		}

		float floatClipThreshold = __half2float(clipThreshold);
		float scale = floatClipThreshold / std::sqrt(sum);

		if (isnan(scale) || isinf(scale)) {
			return 1.0;
		}

		return half(scale);
	}
#endif
#endif
    
    /**
     * calculate the L2Norm of Parameter to void the exploding gradient problem
     */
    T clipGradientScale(std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
        if (parameters.empty()) {
            return T(1.0);
        }

		auto variable   = *parameters.begin();
		auto device     = variable->value.device;
		auto deviceType = device->type;

        if (DeviceType::CPU == deviceType) {
            return clipGradientScaleCPU((static_cast<CPUDevice*>(device))->eigenDevice, parameters, clipThreshold);
        } else {
#ifdef HAVE_CUDA
			return clipGradientScaleGPU(static_cast<GPUDevice*>(device), parameters, clipThreshold);
#else
			DEEP8_RUNTIME_ERROR("does not have a GPU");
#endif
		}
    }

	/**the sub class implement the function*/
    virtual void trainingCPU(Parameter<T> *parameter, T scale) {
    }

    virtual void trainingGPU(Parameter<T> *parameter, T scale) {
    }

public:
    void training(std::unordered_set<Parameter<T>*> &parameters) {
        if (parameters.empty()) {
            return;
        }

        times++;

        T scale(1.0);

        if (clipGradient) {
            scale = clipGradientScale(parameters, clipThreshold);
        }

        for (auto node : parameters) {
            if (!node->updateGradient) {
                continue;
            }

            if (DeviceType::CPU == node->value.device->type) {
                trainingCPU(node, scale);
            } else {
                trainingGPU(node, scale);
            }
        }
    }
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void SGDTrainerKernel(real *gradient, const real scale, const real learningRate, real *value, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		value[i] -= scale * learningRate * gradient[i];
	}
}

#endif

template <typename T>
class SGDTrainer: public Trainer<T> {
public:
    explicit SGDTrainer(T lr = 0.1, bool cg = false, T ct = 5.0): Trainer<T>(lr, cg, ct) {
    }

protected:
    void trainingCPU(Parameter<T> *parameter, T scale) override {
        auto value    = parameter->value;
        auto gradient = parameter->gradient;

        auto device = static_cast<CPUDevice*>(value.device)->eigenDevice;

        eTVec(value).device(*device) -= eTVec(gradient) * (this->learningRate * scale);
    }

	void trainingGPU(Parameter<T> *parameter, T scale) override {
		trainingGPUImpl(parameter, scale);
	}

	template <typename real>
	void trainingGPUImpl(Parameter<real> *parameter, real scale) {
		DEEP8_RUNTIME_ERROR("the type is not support");
	}

#ifdef HAVE_CUDA
	template <>
	void trainingGPUImpl<float>(Parameter<float> *parameter, float scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		float alpha = -1 * (this->learningRate * scale);

		CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
	}

	template <>
	void trainingGPUImpl<double>(Parameter<double> *parameter, double scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		double alpha = -1 * (this->learningRate * scale);

		CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
	}

#ifdef HAVE_HALF
	template <>
	void trainingGPUImpl<half>(Parameter<half> *parameter, half scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		int N = (int) value.size();
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		SGDTrainerKernel<half> << <grideSize, blockSize >> > (gradient.data(), scale, this->learningRate, value.data(), N);
	}
#endif
#endif
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void AdagradTrainerKernel(real *gradient, real scale, real *square, real *value, real epsilon, real learningRate, int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		square[i] += gradient[i] * gradient[i];
		value[i]  -= learningRate * gradient[i] / cuSqrt(square[i] + epsilon);
	}
}

#endif

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
       for (auto item : accumulate) {
           item.second.free();
       }

       accumulate.clear();
    }

protected: 
	template <typename real>
	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<CPUDevice*>(value.device);
		auto eigenDevice = device->eigenDevice;

		if (accumulate.find(parameter) == accumulate.end()) {
			auto ptr = device->malloc(sizeof(real) * gradient.size());

			Tensor<real> square(ptr, gradient.shape, device);
			square.zero();

			accumulate[parameter] = square;
		}

		auto square = accumulate[parameter];

		eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
		eTVec(square).device(*eigenDevice)  += eTVec(gradient).square();
		eTVec(value).device(*eigenDevice)   -= eTVec(gradient) / (eTVec(square) + epsilon).sqrt() * this->learningRate;
	}

#ifdef HAVE_HALF
	template <>
	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif

    void trainingCPU(Parameter<T> *parameter, T scale) override {
		trainingCPUImpl(parameter, scale);
    }

#ifdef HAVE_CUDA
	void trainingGPU(Parameter<T> *parameter, T scale) override {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);
		auto size   = (int) gradient.size();

		if (accumulate.find(parameter) == accumulate.end()) {
			auto ptr = device->malloc(sizeof(T) * size);

			Tensor<T> square(ptr, gradient.shape, device);
			square.zero();

			accumulate[parameter] = square;
		}

		auto square = accumulate[parameter];

		int blockSize = 1024;
		int grideSize = (size + blockSize - 1) / blockSize;

		AdagradTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, square.data(), value.data(), epsilon, learningRate, size);
	}
#endif // HAVE_CUDA

};


#ifdef HAVE_CUDA

template <typename real>
__global__ void AdamTrainerKernel(real *gradient, real scale, real *mt, real *vt, real *value, real beta1, real beta2, real epsilon, real learningRate, int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		mt[i] = mt[i] * beta1 + (1.0 - beta1) * gradient[i];
		vt[i] = vt[i] * beta2 + gradient[i] * gradient[i] * (1.0 - beta2);
		value[i] -= mt[i] / (cuSqrt(vt[i]) + epsilon) * learningRate;
	}
}

#endif

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
   }

   ~AdamTrainer() override {
       for (auto item : m) {
           item.second.free();
       }

       for (auto item : v) {
           item.second.free();
       }

       m.clear();
       v.clear();
   }

protected:

#ifdef HAVE_CUDA
	template <typename real>
	real calculateRealLearningRate(real learningRate, real beta1, real beta2, int64_t times) {
		return learningRate * std::sqrt(1.0 - std::pow(beta2, real(times)) / (1 - std::pow(beta1, real(times));
	}

#ifdef HAVE_HALF
	template <>
	half calculateRealLearningRate<half>(half learningRate, half beta1, half beta2, int64_t times) {
		float learningRateF = __half2float(learningRate);
		float beta1F        = __half2float(beta1);
		float beta2F        = __half2float(beta2);

		return __float2half(calculateRealLearningRate(learningRateF, beta1F, beta2F, times));
	}
#endif

	void trainingGPU(Parameter<T> *parameter, T scale) override {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		if (m.find(parameter) == m.end()) {
			auto ptr = device->malloc(sizeof(T) * gradient.size());
			Tensor<T> mt(ptr, gradient.shape, device);
			mt.zero();

			m[parameter] = mt;
		}

		if (v.find(parameter) == v.end()) {
			auto ptr = device->malloc(sizeof(T) * gradient.size());
			Tensor<T> vt(ptr, gradient.shape, device);
			vt.zero();

			v[parameter] = vt;
		}

		int size = (int) gradient.size();

		auto mt = m[parameter];
		auto vt = v[parameter];

		int blockSize = 1024;
		int grideSize = (size + blockSize - 1) / blockSize;

		auto realLearningRate = calculateRealLearningRate(this->learningRate, beta1, beta2, this->times);

		AdamTrainerKernel<T><<<grideSize, blockSize>>>(gradient.data(), scale, mt.data(), vt.data(), value.data(), beta1, beta2, epsilon, realLearningRate, size);
	}

#endif // HAVE_CUDA

	template <typename real> 
	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
		auto value    = parameter->value;
        auto gradient = parameter->gradient;

		auto device = static_cast<CPUDevice*>(value.device);
		auto eigenDevice = device->eigenDevice;

		if (m.find(parameter) == m.end()) {
			auto ptr = device->malloc(sizeof(real) * gradient.size());
			Tensor<real> mt(ptr, gradient.shape, device);
			mt.zero();

			m[parameter] = mt;
		}

		if (v.find(parameter) == v.end()) {
			auto ptr = device->malloc(sizeof(real) * gradient.size());
			Tensor<real> vt(ptr, gradient.shape, device);
			vt.zero();

			v[parameter] = vt;
		}

		auto mt = m[parameter];
		auto vt = v[parameter];

		eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;

		eTVec(mt).device(*eigenDevice) = eTVec(mt) * beta1 + eTVec(gradient) * (1 - beta1);
		eTVec(vt).device(*eigenDevice) = eTVec(vt) * beta2 + eTVec(gradient).square() * (1 - beta2);

		auto realLearningRate = this->learningRate * sqrt(1 - std::pow(beta2, real(this->times))) / (1 - std::pow(beta1, real(this->times)));

		eTVec(value).device(*eigenDevice) -= eTVec(mt) / (eTVec(vt).sqrt() + epsilon) * realLearningRate;
	}

#ifdef HAVE_HALF
	template <>
	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif

    void trainingCPU(Parameter<T> *parameter, T scale) override {
       trainingCPUImpl(parameter, scale);
    }
};

#ifdef HAVE_CUDA
template <typename real>
__global__ void RMSPropTrainerKernel(real *gradient, real scale, real *vt, real *value, real decay, real epsilon, real learningRate, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		vt[i] = vt[i] * decay + gradient[i] * gradient[i] * (1.0 - decay);
		value[i] -= gradient[i] / cuSqrt(vt[i] + epsilon) * learningRate;
	}
}
#endif

template <typename T>
class RMSPropTrainer: public Trainer<T> {
public:
   T decay;
   T epsilon;

   std::unordered_map<Parameter<T>*, Tensor<T>> v;

   explicit RMSPropTrainer(T learningRate = 0.1, T decay = 0.9, T epsilon = 1e-7, bool clipGradient = false, T clipThreshold = 5.0):
		Trainer<T>(learningRate, clipGradient, clipThreshold), decay(decay), epsilon(epsilon) {
   }

   ~RMSPropTrainer() {
       for (auto item : v) {
           item.second.free();
       }

       v.clear();
   }

protected:

#ifdef HAVE_CUDA
	void trainingGPU(Parameter<T> *parameter, T scale) override {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		if (v.find(parameter) == v.end()) {
			auto ptr = device->malloc(sizeof(T) * gradient.size());
			Tensor<T> vt(ptr, gradient.shape, device);
			vt.zero();

			v[parameter] = vt;
		}

		auto vt = v[parameter];

		int size = (int)gradient.size();

		int blockSize = 1024;
		int grideSize = (size + blockSize - 1) / blockSize;

		RMSPropTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, vt.data(), value.data(), decay, epsilon, learningRate, size);
	}

#endif // HAVE_CUDA

	template <typename real>
	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<CPUDevice*>(value.device);
		auto eigenDevice = device->eigenDevice;

		if (v.find(parameter) == v.end()) {
			auto ptr = device->malloc(sizeof(real) * gradient.size());
			Tensor<real> vt(ptr, gradient.shape, device);
			vt.zero();

			v[parameter] = vt;
		}

		auto vt = v[parameter];

		eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
		eTVec(vt).device(*eigenDevice)       = eTVec(vt) * decay + eTVec(gradient).square() * (1 - decay);
		eTVec(value).device(*eigenDevice)   -= eTVec(gradient) / (eTVec(vt) + epsilon).sqrt() * this->learningRate;
	}

#ifdef HAVE_HALF
	template <>
	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif

    void trainingCPU(Parameter<T> *parameter, T scale) override {
	    trainingCPUImpl(parameter, scale);
   }
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void MomentumTrainerKernel(real *gradient, real scale, real *m, real *value, real alpha, real learningRate, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		m[i] = m[i] * alpha - gradient[i] * learningRate * scale;
		value[i] += m[i];
	}
}
#endif // HAVE_CUDA

template <typename T>
class MomentumTrainer: public Trainer<T> {
public:
   T alpha;

   std::unordered_map<Parameter<T>*, Tensor<T>> momentum;

   explicit MomentumTrainer(T learningRate = 0.1, T alpha = 0.9, bool clipGradient = false, T clipThreshold = 5.0):
		Trainer<T>(learningRate, clipGradient, clipThreshold), alpha(alpha) {
   }

   ~MomentumTrainer() {
       for (auto item : momentum) {
           item.second.free();
       }

       momentum.clear();
   }

protected:

#ifdef HAVE_CUDA

	void trainingGPU(Parameter<T> *parameter, T scale) override {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		if (momentum.find(parameter) == momentum.end()) {
			auto ptr = device->malloc(sizeof(T) * gradient.size());
			Tensor<T> m(ptr, gradient.shape, device);
			m.zero();

			momentum[parameter] = m;
		}

		auto m = momentum[parameter];

		int size = (int)gradient.size();

		int blockSize = 1024;
		int grideSize = (size + blockSize - 1) / blockSize;

		MomentumTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, m.data(), value.data(), alpha, learningRate, size);
	}

#endif // HAVE_CUDA
	template <typename real>
	void trainingCPUImpl(Parameter<real> *parameter, real scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<CPUDevice*>(value.device);
		auto eigenDevice = device->eigenDevice;

		if (momentum.find(parameter) == momentum.end()) {
			auto ptr = device->malloc(sizeof(real) * gradient.size());
			Tensor<real> m(ptr, gradient.shape, device);
			m.zero();

			momentum[parameter] = m;
		}

		auto m = momentum[parameter];

		eTVec(m).device(*eigenDevice) = eTVec(m) * alpha - eTVec(gradient) * this->learningRate * scale;
		eTVec(value).device(*eigenDevice) += eTVec(m);
	}

#ifdef HAVE_HALF
	template <>
	void trainingCPUImpl<half>(Parameter<half> *parameter, half scale) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif

   void trainingCPU(Parameter<T> *parameter, T scale) override {
	   trainingCPUImpl(parameter, scale);
   }
};

}

#endif //DEEP8_TRAINER_H
