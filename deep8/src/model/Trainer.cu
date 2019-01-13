#include "Trainer.h"
#include "GPUDevice.h"
#include "GPUMathUtils.h"

namespace Deep8 {

#ifdef HAVE_CUDA

/**********************************************************************/
/**Trainer*/
/**********************************************************************/

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
#endif

template <>
float Trainer<float>::clipGradientScaleGPU(Executor *executor, Device *d, std::unordered_set<Parameter<float>*> &parameters, float clipThreshold) {
	auto device = (GPUDevice*)d;

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



template <>
double Trainer<double>::clipGradientScaleGPU(Executor *executor, Device *d, std::unordered_set<Parameter<double>*> &parameters, double clipThreshold) {
	auto device = (GPUDevice*)d;

	std::vector<double> l2NormVec;

	for (auto node : parameters) {
		if (!node->updateGradient) {
			continue;
		}

		auto parameter = node;
		auto gradient = parameter->gradient;

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
template <>
half Trainer<half>::clipGradientScaleGPU(Executor *executor, Device *d, std::unordered_set<Parameter<half>*> &parameters, half clipThreshold) {
	auto device = (GPUDevice*)d;

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
		auto gradient = parameter->gradient;

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

template <typename T>
Tensor<T> Trainer<T>::createTensorGPU(Device *device, Shape &shape) {
	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device->malloc(storageSize);
	auto refPtr = (size_t*)device->mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, device);

	return Tensor<T>(storage, 0, shape);
}

template Tensor<float> Trainer<float>::createTensorGPU(Device *device, Shape &shape);
template Tensor<double> Trainer<double>::createTensorGPU(Device *device, Shape &shape);
#ifdef HAVE_HALF
template Tensor<half> Trainer<half>::createTensorGPU(Device *device, Shape &shape);
#endif


/**********************************************************************/
/**SGDTrainer*/
/**********************************************************************/
#ifdef HAVE_CUDA
template <typename real>
__global__ void SGDTrainerKernel(real *gradient, const real scale, const real learningRate, real *value, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		value[i] -= scale * learningRate * gradient[i];
	}
}
#endif

template <>
void SGDTrainer<float>::updateGPU(Executor *executor, Parameter<float> *parameter, int64_t steps, float learningRate, float scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());

	float alpha = -1 * (learningRate * scale);

	CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
}

template <>
void SGDTrainer<double>::updateGPU(Executor *executor, Parameter<double> *parameter, int64_t steps, double learningRate, double scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());

	double alpha = -1 * (learningRate * scale);

	CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
}

#ifdef HAVE_HALF
template <>
void SGDTrainer<half>::updateGPU(Executor *executor, Parameter<half> *parameter, int64_t steps, half learningRate, half scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	int N = (int)value.size();
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	SGDTrainerKernel<half> << <grideSize, blockSize >> > (gradient.data(), scale, learningRate, value.data(), N);
}
#endif

/**********************************************************************/
/**AdagradTrainer*/
/**********************************************************************/
#ifdef HAVE_CUDA
template <typename real>
__global__ void AdagradTrainerKernel(real *gradient, real scale, real *square, real *value, real epsilon, real learningRate, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		square[i] += gradient[i] * gradient[i];
		value[i] -= learningRate * gradient[i] / CuMath::cuSqrt(square[i] + epsilon);
	}
}
#endif

template <typename T>
void AdagradTrainer<T>::trainingGPU(Parameter<T> *parameter, T scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());
	auto size = (int)gradient.size();

	if (accumulate.find(parameter) == accumulate.end()) {
		auto square = createTensorGPU(device, gradient.shape);
		square.zero();

		accumulate[parameter] = square;
	}

	auto square = accumulate[parameter];

	int blockSize = 1024;
	int grideSize = (size + blockSize - 1) / blockSize;

	AdagradTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, square.data(), value.data(), epsilon, learningRate, size);
}

template void AdagradTrainer<float>::trainingGPU(Parameter<float> *parameter, float scale);
template void AdagradTrainer<double>::trainingGPU(Parameter<double> *parameter, double scale);
#ifdef HAVE_HALF
template void AdagradTrainer<half>::trainingGPU(Parameter<half> *parameter, half scale);
#endif

/**********************************************************************/
/**AdamTrainer*/
/**********************************************************************/
#ifdef HAVE_CUDA
template <typename real>
__global__ void AdamTrainerKernel(real *gradient, real scale, real *mt, real *vt, real *value, real beta1, real beta2, real epsilon, real learningRate, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		mt[i] = mt[i] * beta1 + (real(1.0) - beta1) * gradient[i];
		vt[i] = vt[i] * beta2 + gradient[i] * gradient[i] * (real(1.0) - beta2);
		value[i] -= mt[i] / (CuMath::cuSqrt(vt[i]) + epsilon) * learningRate;
	}
}
#endif

template <typename T>
T AdamTrainer<T>::calculateRealLearningRate(T learningRate, T beta1, T beta2, int64_t times) {
	return learningRate * std::sqrt(1.0 - std::pow(beta2, T(times))) / (1 - std::pow(beta1, T(times)));
}

#ifdef HAVE_HALF
template <>
half AdamTrainer<half>::calculateRealLearningRate(half learningRate, half beta1, half beta2, int64_t times) {
	float learningRateF = __half2float(learningRate);
	float beta1F = __half2float(beta1);
	float beta2F = __half2float(beta2);

	return __float2half(calculateRealLearningRate(learningRateF, beta1F, beta2F, times));
}
#endif

template <typename T>
void AdamTrainer<T>::trainingGPU(Parameter<T> *parameter, T scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());

	if (m.find(parameter) == m.end()) {
		auto mt = createTensorGPU(device, gradient.shape);
		mt.zero();

		m[parameter] = mt;
	}

	if (v.find(parameter) == v.end()) {
		auto vt = createTensorGPU(device, gradient.shape);
		vt.zero();

		v[parameter] = vt;
	}

	int size = (int)gradient.size();

	auto mt = m[parameter];
	auto vt = v[parameter];

	int blockSize = 1024;
	int grideSize = (size + blockSize - 1) / blockSize;

	auto realLearningRate = calculateRealLearningRate(this->learningRate, beta1, beta2, this->times);

	AdamTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, mt.data(), vt.data(), value.data(), beta1, beta2, epsilon, realLearningRate, size);
}

template void AdamTrainer<float>::trainingGPU(Parameter<float> *parameter, float scale);
template void AdamTrainer<double>::trainingGPU(Parameter<double> *parameter, double scale);
#ifdef HAVE_HALF
template void AdamTrainer<half>::trainingGPU(Parameter<half> *parameter, half scale);
#endif

/**********************************************************************/
/**RMSPropTrainer*/
/**********************************************************************/
#ifdef HAVE_CUDA
template <typename real>
__global__ void RMSPropTrainerKernel(real *gradient, real scale, real *vt, real *value, real decay, real epsilon, real learningRate, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		gradient[i] *= scale;
		vt[i] = vt[i] * decay + gradient[i] * gradient[i] * (real(1.0) - decay);
		value[i] -= gradient[i] / CuMath::cuSqrt(vt[i] + epsilon) * learningRate;
	}
}
#endif

template <typename T>
void RMSPropTrainer<T>::trainingGPU(Parameter<T> *parameter, T scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());

	if (v.find(parameter) == v.end()) {
		auto vt = createTensorGPU(device, gradient.shape);
		vt.zero();

		v[parameter] = vt;
	}

	auto vt = v[parameter];

	int size = (int)gradient.size();

	int blockSize = 1024;
	int grideSize = (size + blockSize - 1) / blockSize;

	RMSPropTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, vt.data(), value.data(), decay, epsilon, learningRate, size);
}

template void RMSPropTrainer<float>::trainingGPU(Parameter<float> *parameter, float scale);
template void RMSPropTrainer<double>::trainingGPU(Parameter<double> *parameter, double scale);
#ifdef HAVE_HALF
template void RMSPropTrainer<half>::trainingGPU(Parameter<half> *parameter, half scale);
#endif


/**********************************************************************/
/**MomentumTrainer*/
/**********************************************************************/
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
#endif


template <typename T>
void MomentumTrainer<T>::trainingGPU(Parameter<T> *parameter, T scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());

	if (momentum.find(parameter) == momentum.end()) {
		auto m = createTensorGPU(device, gradient.shape);
		m.zero();

		momentum[parameter] = m;
	}

	auto m = momentum[parameter];

	int size = (int)gradient.size();

	int blockSize = 1024;
	int grideSize = (size + blockSize - 1) / blockSize;

	MomentumTrainerKernel<T> << <grideSize, blockSize >> > (gradient.data(), scale, m.data(), value.data(), alpha, learningRate, size);
}

template void MomentumTrainer<float>::trainingGPU(Parameter<float> *parameter, float scale);
template void MomentumTrainer<double>::trainingGPU(Parameter<double> *parameter, double scale);
#ifdef HAVE_HALF
template void MomentumTrainer<half>::trainingGPU(Parameter<half> *parameter, half scale);
#endif

#endif

}