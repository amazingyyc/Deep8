#include "TensorInit.h"
#include "GPUDevice.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void TensorInitConstantKernel(real *value, real scalar, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		value[i] = scalar;
	}
}

template <typename real>
__global__ void TensorInitPositiveUnitballKernel(real *value, real sum, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		value[i] /= sum;
	}
}

#ifdef HAVE_HALF
__global__ void TensorInitConvertFloatToHalf(const float *from, half* to, int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		to[i] = __float2half(from[i]);
	}
}
#endif


template <>
void TensorInit<float>::constantGPU(Tensor<float> &tensor, float v) {
	int N = (int)tensor.size();

	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	TensorInitConstantKernel<float> << <grideSize, blockSize >> > (tensor.data(), v, N);
}

template <>
void TensorInit<double>::constantGPU(Tensor<double> &tensor, double v) {
	int N = (int)tensor.size();

	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	TensorInitConstantKernel<double> << <grideSize, blockSize >> > (tensor.data(), v, N);
}

#ifdef HAVE_HALF
template <>
void TensorInit<half>::constantGPU(Tensor<half> &tensor, half v) {
	int N = (int)tensor.size();

	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	TensorInitConstantKernel<half> << <grideSize, blockSize >> > (tensor.data(), v, N);
}
#endif

template <>
void TensorInit<float>::uniformGPU(Tensor<float> &tensor) {
	auto device = static_cast<GPUDevice*>(tensor.device());
	CURAND_CHECK(curandGenerateUniform(device->curandGenerator, tensor.data(), (size_t)tensor.size()));
}

template <>
void TensorInit<double>::uniformGPU(Tensor<double> &tensor) {
	auto device = static_cast<GPUDevice*>(tensor.device());
	CURAND_CHECK(curandGenerateUniformDouble(device->curandGenerator, tensor.data(), (size_t)tensor.size()));
}

#ifdef HAVE_HALF
template <>
void TensorInit<half>::uniformGPU(Tensor<half> &tensor) {
	auto device = static_cast<GPUDevice*>(tensor.device());

	auto size = (size_t)tensor.size();
	auto ptr = (float*)device->malloc(sizeof(float) * size);

	CURAND_CHECK(curandGenerateUniform(device->curandGenerator, ptr, size));

	int N = (int)size;
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	TensorInitConvertFloatToHalf << <grideSize, blockSize >> > (ptr, tensor.data(), N);

	device->free(ptr);
}
#endif

template <>
void TensorInit<float>::gaussianGPU(Tensor<float> &tensor, float mean, float stddev) {
	auto device = static_cast<GPUDevice*>(tensor.device());

	CURAND_CHECK(curandGenerateNormal(device->curandGenerator, tensor.data(), (size_t)tensor.size(), mean, stddev));
}

template <>
void TensorInit<double>::gaussianGPU(Tensor<double> &tensor, double mean, double stddev) {
	auto device = static_cast<GPUDevice*>(tensor.device());

	CURAND_CHECK(curandGenerateNormalDouble(device->curandGenerator, tensor.data(), (size_t)tensor.size(), mean, stddev));
}

#ifdef HAVE_HALF
template <>
void TensorInit<half>::gaussianGPU(Tensor<half> &tensor, half mean, half stddev) {
	auto device = static_cast<GPUDevice*>(tensor.device());

	auto size = (size_t)tensor.size();
	auto ptr  = (float*)device->malloc(sizeof(float) * size);

	CURAND_CHECK(curandGenerateNormal(device->curandGenerator, ptr, size, __half2float(mean), __half2float(stddev)));

	int N = (int)size;
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	TensorInitConvertFloatToHalf << <grideSize, blockSize >> > (ptr, tensor.data(), N);

	device->free(ptr);
}
#endif

template <>
void TensorInit<float>::positiveUnitballGPU(Tensor<float> &tensor) {
	auto device = static_cast<GPUDevice*>(tensor.device());
	int N = (int)tensor.size();

	uniformGPU(tensor);

	float sum = 0;

	CUBLAS_CHECK(cublasSasum(device->cublasHandle, N, tensor.data(), 1, &sum));

	if (0 != sum) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TensorInitPositiveUnitballKernel<float> << <grideSize, blockSize >> > (tensor.data(), sum, N);
	}
}

template <>
void TensorInit<double>::positiveUnitballGPU(Tensor<double> &tensor) {
	auto device = static_cast<GPUDevice*>(tensor.device());
	int N = (int)tensor.size();

	uniformGPU(tensor);

	double sum = 0;

	CUBLAS_CHECK(cublasDasum(device->cublasHandle, N, tensor.data(), 1, &sum));

	if (0 != sum) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TensorInitPositiveUnitballKernel<double> << <grideSize, blockSize >> > (tensor.data(), sum, N);
	}
}

#ifdef HAVE_HALF
void TensorInit<half>::positiveUnitballGPU(Tensor<half> &tensor) {
	auto device = static_cast<GPUDevice*>(tensor.device());

	auto size = (size_t)tensor.size();
	auto ptr = (float*)device->malloc(sizeof(float) * size);
	
	CURAND_CHECK(curandGenerateUniform(device->curandGenerator, ptr, size));

	float sum = 0;

	CUBLAS_CHECK(cublasSasum(device->cublasHandle, (int) size, ptr, 1, &sum));

	int N = (int)size;
	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	if (0 != sum) {
		TensorInitPositiveUnitballKernel<float> << <grideSize, blockSize >> > (ptr, sum, N);
	}

	TensorInitConvertFloatToHalf << <grideSize, blockSize >> > (ptr, tensor.data(), N);

	device->free(ptr);
}
#endif

#endif

}