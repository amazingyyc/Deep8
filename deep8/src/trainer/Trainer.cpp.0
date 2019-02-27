#include "Trainer.h"

namespace Deep8 {


/**********************************************************************/
/**Trainer*/
/**********************************************************************/
template <typename T>
Trainer<T>::Trainer(LearningRateIterator *lr, bool cg = false, T ct = 5.0):
		learningRateIterator(lr), clipGradient(cg), clipThreshold(ct) {
}

/**
 * calculate the L2Norm of Parameter to void the exploding gradient problem
 */
template <typename T>
T Trainer<T>::clipGradientScale(Executor *executor, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
	if (parameters.empty()) {
		return T(1.0);
	}

	auto variable   = *parameters.begin();
	auto device     = variable->value.device();

	if (DeviceType::CPU == device->type) {
		return clipGradientScaleCPU(executor, ((CPUDevice*)device)->eigenDevice, parameters, clipThreshold);
	} else {
#ifdef HAVE_CUDA
		return clipGradientScaleGPU(executor, device, parameters, clipThreshold);
#else
		DEEP8_RUNTIME_ERROR("does not have a GPU");
#endif
	}
}

template <typename T>
T Trainer<T>::clipGradientScaleCPU(Executor *executor,  Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
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

	auto scale = clipThreshold / std::sqrt(sum);

	if (isnan(scale) || isinf(scale)) {
		return T(1.0);
	}

	return scale;
}

#ifdef HAVE_HALF
template <>
half Trainer<half>::clipGradientScaleCPU(Executor *executor,  Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

template <typename T>
Tensor<T> Trainer<T>::createTensorCPU(Device *device, Shape &shape) {
	auto storageSize = sizeof(T) * shape.size();

	auto ptr    = device->malloc(storageSize);
	auto refPtr = (size_t*)device->malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, device);

	return Tensor<T>(storage, 0, shape);
}

/**update the parameter*/
template <typename T>
void Trainer<T>::updateCPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) {
	DEEP8_RUNTIME_ERROR("can not call this function in Trainer");
}

#ifdef HAVE_CUDA
template <typename T>
void Trainer<T>::updateGPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) {
	DEEP8_RUNTIME_ERROR("can not call this function in Trainer");
}
#endif

template <typename T>
void Trainer<T>::update(Executor *executor, std::unordered_set<Parameter<T>*> &parameters, int64_t steps) {
	if (parameters.empty()) {
		return;
	}

	T scale(1.0);

	if (clipGradient) {
		scale = clipGradientScale(executor, parameters, clipThreshold);
	}

	T learningRate = learningRateIterator->generateLearningRate(steps);

	for (auto parameter : parameters) {
		if (!parameter->updateGradient) {
			continue;
		}

		if (DeviceType::CPU == parameter->value.device()->type) {
			updateCPU(executor, parameter, steps, learningRate, scale);
		} else {
#ifdef HAVE_CUDA
			updateGPU(executor, parameter, steps, learningRate, scale);
#else
			DEEP8_RUNTIME_ERROR("does not have a GPU");
#endif
		}
	}
}

DEEP8_DECLARATION_INSTANCE(Trainer)

/**********************************************************************/
/**SGDTrainer*/
/**********************************************************************/
template <typename T>
SGDTrainer<T>::SGDTrainer(LearningRateIterator *lr, bool cg = false, T ct = 5.0): Trainer<T>(lr, cg, ct) {
}

template <typename T>
void SGDTrainer<T>::updateCPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<CPUDevice*>(value.device())->eigenDevice;

	eTVec(value).device(*device) -= eTVec(gradient) * (learningRate * scale);
}

#ifdef HAVE_HALF
template <>
void SGDTrainer<half>::updateCPU(Executor *executor, Parameter<T> *parameter, int64_t steps, T learningRate, T scale) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

DEEP8_DECLARATION_INSTANCE(SGDTrainer)

/**********************************************************************/
/**AdagradTrainer*/
/**********************************************************************/
template <typename T>
AdagradTrainer<T>::AdagradTrainer(LearningRateIterator *learningRate, T epsilon, bool clipGradient, T clipThreshold)
		:Trainer<T>(learningRate, clipGradient, clipThreshold), epsilon(epsilon) {
		check(epsilon);
}

template <typename T>
void AdagradTrainer<T>::check(T epsilon) {
	DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
}

#ifdef HAVE_HALF
template <>
void AdagradTrainer<half>::check(half epsilon) {
	DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
}
#endif

template <typename T>
void AdagradTrainer<T>::trainingCPU(Parameter<T> *parameter, T scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<CPUDevice*>(value.device());
	auto eigenDevice = device->eigenDevice;

	if (accumulate.find(parameter) == accumulate.end()) {
		auto square = this->createTensorCPU(device, gradient.shape);
		square.zero();

		accumulate[parameter] = square;
	}

	auto square = accumulate[parameter];

	eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
	eTVec(square).device(*eigenDevice) += eTVec(gradient).square();
	eTVec(value).device(*eigenDevice) -= eTVec(gradient) / (eTVec(square) + epsilon).sqrt() * this->learningRate;
}

#ifdef HAVE_HALF
template <>
void AdagradTrainer<half>::trainingCPU(Parameter<half> *parameter, half scale) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

DEEP8_DECLARATION_INSTANCE(AdagradTrainer)


/**********************************************************************/
/**AdamTrainer*/
/**********************************************************************/
template <typename T>
AdamTrainer<T>::AdamTrainer(T learningRate, T beta1, T beta2, T epsilon, bool clipGradient, T clipThreshold):
		Trainer<T>(learningRate, clipGradient, clipThreshold), beta1(beta1), beta2(beta2), epsilon(epsilon) {
	check(epsilon);
}

template <typename T>
void AdamTrainer<T>::check(T epsilon) {
	DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
}

#ifdef HAVE_HALF
template <>
   void AdamTrainer<half>::check(half epsilon) {
	   DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
   }
#endif

template <typename T>
AdamTrainer<T>::~AdamTrainer() {
	m.clear();
	v.clear();
}

template <typename T>
void AdamTrainer<T>::trainingCPU(Parameter<T> *parameter, T scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<CPUDevice*>(value.device());
	auto eigenDevice = device->eigenDevice;

	if (m.find(parameter) == m.end()) {
		auto mt = this->createTensorCPU(device, gradient.shape);
		mt.zero();

		m[parameter] = mt;
	}

	if (v.find(parameter) == v.end()) {
		auto vt = this->createTensorCPU(device, gradient.shape);
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

#ifdef HAVE_HALF
template <>
void AdamTrainer<half>::trainingCPU(Parameter<half> *parameter, half scale) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

DEEP8_DECLARATION_INSTANCE(AdamTrainer)


/**********************************************************************/
/**RMSPropTrainer*/
/**********************************************************************/
template <typename T>
RMSPropTrainer<T>::RMSPropTrainer(T learningRate, T decay, T epsilon, bool clipGradient, T clipThreshold):
		Trainer<T>(learningRate, clipGradient, clipThreshold), decay(decay), epsilon(epsilon) {
	check(epsilon);
}

template <typename T>
void RMSPropTrainer<T>::check(T epsilon) {
	DEEP8_ARGUMENT_CHECK(0 != epsilon, "epsilon can not be 0");
}

#ifdef HAVE_HALF
template <>
void RMSPropTrainer<half>::check(half epsilon) {
   DEEP8_ARGUMENT_CHECK(0 != __half2float(epsilon), "epsilon can not be 0");
}
#endif

template <typename T>
RMSPropTrainer<T>::~RMSPropTrainer() {
	v.clear();
}

template <typename T>
void RMSPropTrainer<T>::trainingCPU(Parameter<T> *parameter, T scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<CPUDevice*>(value.device());
	auto eigenDevice = device->eigenDevice;

	if (v.find(parameter) == v.end()) {
		auto vt = this->createTensorCPU(device, gradient.shape);
		vt.zero();

		v[parameter] = vt;
	}

	auto vt = v[parameter];

	eTVec(gradient).device(*eigenDevice) = eTVec(gradient) * scale;
	eTVec(vt).device(*eigenDevice) = eTVec(vt) * decay + eTVec(gradient).square() * (1 - decay);
	eTVec(value).device(*eigenDevice) -= eTVec(gradient) / (eTVec(vt) + epsilon).sqrt() * this->learningRate;
}

#ifdef HAVE_HALF
template <>
void RMSPropTrainer<half>::trainingCPU(Parameter<half> *parameter, half scale) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

DEEP8_DECLARATION_INSTANCE(RMSPropTrainer)

/**********************************************************************/
/**MomentumTrainer*/
/**********************************************************************/
template <typename T>
MomentumTrainer<T>::MomentumTrainer(T learningRate, T alpha, bool clipGradient, T clipThreshold):
		Trainer<T>(learningRate, clipGradient, clipThreshold), alpha(alpha) {
}

template <typename T>
MomentumTrainer<T>::~MomentumTrainer() {
	momentum.clear();
}

template <typename T>
void MomentumTrainer<T>::trainingCPU(Parameter<T> *parameter, T scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<CPUDevice*>(value.device());
	auto eigenDevice = device->eigenDevice;

	if (momentum.find(parameter) == momentum.end()) {
		auto m = this->createTensorCPU(device, gradient.shape);
		m.zero();

		momentum[parameter] = m;
	}

	auto m = momentum[parameter];

	eTVec(m).device(*eigenDevice) = eTVec(m) * alpha - eTVec(gradient) * this->learningRate * scale;
	eTVec(value).device(*eigenDevice) += eTVec(m);
}

#ifdef HAVE_HALF
template <>
void MomentumTrainer<half>::trainingCPU(Parameter<half> *parameter, half scale) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

DEEP8_DECLARATION_INSTANCE(MomentumTrainer)

}