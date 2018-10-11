#include "Trainer.h"
#include "../basic/GPUDevice.h"

namespace Deep8 {

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
float Trainer<float>::clipGradientScaleGPU(Device *d, std::unordered_set<Parameter<float>*> &parameters, float clipThreshold) {
#ifdef HAVE_CUDA
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
#else
	DEEP8_RUNTIME_ERROR("does not have a GPU");
#endif
}

template <>
double Trainer<double>::clipGradientScaleGPU(Device *d, std::unordered_set<Parameter<double>*> &parameters, double clipThreshold) {
#ifdef HAVE_CUDA
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

#else
DEEP8_RUNTIME_ERROR("does not have a GPU");
#endif
}

#ifdef HAVE_HALF
template <>
half Trainer<half>::clipGradientScaleGPU(Device *d, std::unordered_set<Parameter<half>*> &parameters, half clipThreshold) {
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
void SGDTrainer<float>::trainingGPU(Parameter<float> *parameter, float scale) {
	auto value    = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());

	float alpha = -1 * (this->learningRate * scale);

	CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
}

template <>
void SGDTrainer<double>::trainingGPU(Parameter<double> *parameter, double scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	auto device = static_cast<GPUDevice*>(value.device());

	double alpha = -1 * (this->learningRate * scale);

	CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)value.size(), &alpha, gradient.data(), 1, value.data(), 1));
}

#ifdef HAVE_HALF
template <>
void SGDTrainer<half>::trainingGPU(Parameter<half> *parameter, half scale) {
	auto value = parameter->value;
	auto gradient = parameter->gradient;

	int N = (int)value.size();
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	SGDTrainerKernel<half> << <grideSize, blockSize >> > (gradient.data(), scale, this->learningRate, value.data(), N);
}
#endif


}