#include "Trainer.h"

namespace Deep8 {


/**********************************************************************/
/**Trainer*/
/**********************************************************************/

template <typename T>
T Trainer<T>::clipGradientScaleCPU(Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
	std::vector<T> l2NormVec;

	for (auto node : parameters) {
		if (!node->updateGradient) {
			continue;
		}

		auto parameter = node;
		auto gradient = parameter->gradient;

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
		return T(1);
	}

	return scale;
}

#ifdef HAVE_HALF
template <>
half Trainer<half>::clipGradientScaleCPU(Eigen::ThreadPoolDevice *device, std::unordered_set<Parameter<half>*> &parameters, half clipThreshold) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

/**
 * calculate the L2Norm of Parameter to void the exploding gradient problem
 */
template <typename T>
T Trainer<T>::clipGradientScale(std::unordered_set<Parameter<T>*> &parameters, T clipThreshold) {
	if (parameters.empty()) {
		return T(1.0);
	}

	auto variable   = *parameters.begin();
	auto device     = variable->value.device();
	auto deviceType = device->type;

	if (DeviceType::CPU == deviceType) {
		return clipGradientScaleCPU((static_cast<CPUDevice*>(device))->eigenDevice, parameters, clipThreshold);
	} else {
#ifdef HAVE_CUDA
		return clipGradientScaleGPU(device, parameters, clipThreshold);
#else
		DEEP8_RUNTIME_ERROR("does not have a GPU");
#endif
	}
}

template <typename T>
Tensor<T> Trainer<T>::createTensorCPU(Device *device, Shape &shape) {
	auto storageSize = sizeof(T) * shape.size();

	auto ptr    = device->malloc(storageSize);
	auto refPtr = (size_t*)device->malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> Trainer<T>::createTensorGPU(Device *device, Shape &shape) {
#ifdef HAVE_CUDA
	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device->malloc(storageSize);
	auto refPtr = (size_t*)device->mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, device);

	return Tensor<T>(storage, 0, shape);
#else
	DEEP8_RUNTIME_ERROR("does not have a GPU");
#endif
}

template <typename T>
void Trainer<T>::training(std::unordered_set<Parameter<T>*> &parameters) {
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

		if (DeviceType::CPU == node->value.device()->type) {
			trainingCPU(node, scale);
		} else {
			trainingGPU(node, scale);
		}
	}
}

DEEP8_DECLARATION_INSTANCE(Trainer)

/**********************************************************************/
/**SGDTrainer*/
/**********************************************************************/
template <typename T>
void SGDTrainer<T>::trainingCPU(Parameter<T> *parameter, T scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<CPUDevice*>(value.device())->eigenDevice;

	eTVec(value).device(*device) -= eTVec(gradient) * (this->learningRate * scale);
}

DEEP8_DECLARATION_INSTANCE(SGDTrainer)



}