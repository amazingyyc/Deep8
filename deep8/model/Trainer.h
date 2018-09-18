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

    T clipGradientScaleCPU(Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
        std::vector<T> l2NormVec;

        for (auto node : parameters) {
            if (!node->updateGradient) {
                continue;
            }

			auto parameter = node;
			auto gradient  = parameter->gradient;

            l2NormVec.push_back(T(0));

            Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>> sum(static_cast<T*>(&(l2NormVec[l2NormVec.size() - 1])));
            Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> vec(gradient.data(), gradient.size());

            sum.device(*device) = vec.square().sum();
        }

        T sum = 0;

        for (auto item : l2NormVec) {
            sum += item;
        }

        auto scale = clipThreshold / sqrt(sum);

        if (isnan(scale) || isinf(scale)) {
            return T(1);
        }

        return scale;
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
#endif
    
    /**
     * calculate the L2Norm of Parameter to void the exploding gradient problem
     */
    T clipGradientScale(std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
        if (parameters.empty()) {
            return T(1);
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

        T scale = 1;

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

#ifdef HAVE_CUDA

	void trainingGPU(Parameter<float> *parameter, float scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		float alpha = -1 * (this->learningRate * scale);

		CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
	}

	void trainingGPU(Parameter<double> *parameter, double scale) {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		double alpha = -1 * (this->learningRate * scale);

		CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
	}

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

    explicit AdagradTrainer(T learningRate = 0.1, T epsilon = 1e-8, bool clipGradient = false, T clipThreshold = 5.0)
            :Trainer<T>(learningRate, clipGradient, clipThreshold), epsilon(epsilon) {
        DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
    }

    ~AdagradTrainer() override {
       for (auto item : accumulate) {
           item.second.free();
       }

       accumulate.clear();
    }

protected:
    void trainingCPU(Parameter<T> *parameter, T scale) override {
        auto value    = parameter->value;
        auto gradient = parameter->gradient;

        auto device = static_cast<CPUDevice*>(value.device);
        auto eigenDevice = device->eigenDevice;

        if (accumulate.find(parameter) == accumulate.end()) {
            auto ptr = device->malloc(sizeof(T) * gradient.size());

			Tensor<T> square(ptr, gradient.shape, device);
            square.zero();

			accumulate[parameter] = square;
        }

        auto square = accumulate[parameter];

        eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
        eTVec(square).device(*eigenDevice)  += eTVec(gradient).square();
        eTVec(value).device(*eigenDevice)   -= eTVec(gradient) / (eTVec(square) + epsilon).sqrt() * this->learningRate;
    }

#ifdef HAVE_CUDA

	void trainingGPU(Parameter<T> *parameter, T scale) override {
		auto value    = parameter->value;
		auto gradient = parameter->gradient;

		auto device = static_cast<GPUDevice*>(value.device);

		int size = (int)gradient.size();

		if (accumulate.find(parameter) == accumulate.end()) {
			auto ptr = device->malloc(sizeof(T) * size);

			Tensor<T> square(ptr, gradient.shape, device);
			square.zero();

			accumulate[parameter] = square;
		}

		auto square = accumulate[parameter];

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AdagradTrainerKernel<T>, 0, size));

		grideSize = (size + blockSize - 1) / blockSize;

		AdagradTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, square.data(), value.data(), epsilon, learningRate, size);
	}

#endif // HAVE_CUDA

};


#ifdef HAVE_CUDA

template <typename real>
__global__ void AdamTrainerKernel(real *gradient, real scale, real *mt, real *vt, real *value, real beta1, real beta2, real epsilon, real learningRate, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		mt[i] = mt[i] * beta1 + (1 - beta1) * gradient[i];
		vt[i] = vt[i] * beta2 + gradient[i] * gradient[i] * (1 - beta2);
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

    explicit AdamTrainer(T learningRate = 0.1, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8, bool clipGradient = false, T clipThreshold = 5.0):
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

	void trainingGPU(Parameter<T> *parameter, T scale) override {
		auto value = parameter->value;
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

		int size = (int)gradient.size();

		auto mt = m[parameter];
		auto vt = v[parameter];

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AdamTrainerKernel<T>, 0, size));

		grideSize = (size + blockSize - 1) / blockSize;

		T realLearningRate = this->learningRate * std::sqrt(1 - std::pow(beta2, T(this->times))) / (1 - std::pow(beta1, T(this->times)));

		AdamTrainerKernel<T><<<grideSize, blockSize>>>(gradient.data(), scale, mt.data(), vt.data(), value.data(), beta1, beta2, epsilon, realLearningRate, size);
	}

#endif // HAVE_CUDA


    void trainingCPU(Parameter<T> *parameter, T scale) override {
        auto value    = parameter->value;
        auto gradient = parameter->gradient;

        auto device = static_cast<CPUDevice*>(value.device);
        auto eigenDevice = device->eigenDevice;

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

        auto mt = m[parameter];
        auto vt = v[parameter];

        eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;

        eTVec(mt).device(*eigenDevice) = eTVec(mt) * beta1 + eTVec(gradient) * (1 - beta1);
        eTVec(vt).device(*eigenDevice) = eTVec(vt) * beta2 + eTVec(gradient).square() * (1 - beta2);

        auto realLearningRate = this->learningRate * sqrt(1 - std::pow(beta2, T(this->times))) / (1 - std::pow(beta1, T(this->times)));

        eTVec(value).device(*eigenDevice) -= eTVec(mt) / (eTVec(vt).sqrt() + epsilon) * realLearningRate;
    }
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void RMSPropTrainerKernel(real *gradient, real scale, real *vt, real *value, real decay, real epsilon, real learningRate, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		vt[i] = vt[i] * decay + gradient[i] * gradient[i] * (1 - decay);
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

    explicit RMSPropTrainer(T learningRate = 0.1, T decay = 0.9, T epsilon = 1e-8, bool clipGradient = false, T clipThreshold = 5.0):
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
		auto value = parameter->value;
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

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, RMSPropTrainerKernel<T>, 0, size));

		grideSize = (size + blockSize - 1) / blockSize;

		RMSPropTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, vt.data(), value.data(), decay, epsilon, learningRate, size);
	}

#endif // HAVE_CUDA


    void trainingCPU(Parameter<T> *parameter, T scale) override {
        auto value    = parameter->value;
        auto gradient = parameter->gradient;

        auto device = static_cast<CPUDevice*>(value.device);
        auto eigenDevice = device->eigenDevice;

        if (v.find(parameter) == v.end()) {
            auto ptr = device->malloc(sizeof(T) * gradient.size());
            Tensor<T> vt(ptr, gradient.shape, device);
            vt.zero();

			v[parameter] = vt;
        }

        auto vt = v[parameter];

        eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
        eTVec(vt).device(*eigenDevice) = eTVec(vt) * decay + eTVec(gradient).square() * (1 - decay);
        eTVec(value).device(*eigenDevice) -= eTVec(gradient) / (eTVec(vt) + epsilon).sqrt() * this->learningRate;
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
		auto value = parameter->value;
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

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MomentumTrainerKernel<T>, 0, size));

		grideSize = (size + blockSize - 1) / blockSize;

		MomentumTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, m.data(), value.data(), alpha, learningRate, size);
	}

#endif // HAVE_CUDA


    void trainingCPU(Parameter<T> *parameter, T scale) override {
        auto value    = parameter->value;
        auto gradient = parameter->gradient;

        auto device = static_cast<CPUDevice*>(value.device);
        auto eigenDevice = device->eigenDevice;

        if (momentum.find(parameter) == momentum.end()) {
            auto ptr = device->malloc(sizeof(T) * gradient.size());
            Tensor<T> m(ptr, gradient.shape, device);
            m.zero();

			momentum[parameter] = m;
        }

        auto m = momentum[parameter];

        eTVec(m).device(*eigenDevice) = eTVec(m) * alpha - eTVec(gradient) * this->learningRate * scale;
        eTVec(value).device(*eigenDevice) += eTVec(m);
    }
};

}

#endif //DEEP8_TRAINER_H
