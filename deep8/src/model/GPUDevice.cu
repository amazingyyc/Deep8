#include "GPUDevice.h"

namespace Deep8 {

#ifdef HAVE_CUDA

GPUDevice::GPUDevice() : GPUDevice(0) {
}
 
GPUDevice::GPUDevice(int id) : Device(DeviceType::GPU), deviceId(id) {
	CUDA_CHECK(cudaSetDevice(deviceId));

	memoryPool = new GPUMemoryPool(deviceId);

	CUBLAS_CHECK(cublasCreate(&cublasHandle));

	CURAND_CHECK(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGenerator, (unsigned long long)time(nullptr)));

#ifdef HAVE_CUDNN
	CUDNN_CHECK(cudnnCreate(&cudnnHandle));
#endif

#ifdef HAVE_HALF
	void *ptr = memoryPool->malloc(sizeof(float) * 3 + sizeof(double) * 3 + sizeof(half) * 3);

	oneFloat = (float*)ptr;
	zeroFloat = oneFloat + 1;
	minusOneFloat = zeroFloat + 1;

	oneDouble = (double*)(minusOneFloat + 1);
	zeroDouble = oneDouble + 1;
	minusOneDouble = zeroDouble + 1;

	oneHalf = (half*)(minusOneDouble + 1);
	zeroHalf = oneHalf + 1;
	minusOneHalf = zeroHalf + 1;

	float  numberF[3] = { 1, 0, -1 };
	double numberD[3] = { 1, 0, -1 };
	half   numberH[3] = { 1.0, 0.0, -1.0 };

	memoryPool->copyFromCPUToGPU(&numberF[0], oneFloat, sizeof(float) * 3);
	memoryPool->copyFromCPUToGPU(&numberD[0], oneDouble, sizeof(double) * 3);
	memoryPool->copyFromCPUToGPU(&numberH[0], oneHalf, sizeof(half) * 3);

#else
	void *ptr = memoryPool->malloc(sizeof(float) * 3 + sizeof(double) * 3);

	oneFloat = (float*)ptr;
	zeroFloat = oneFloat + 1;
	minusOneFloat = zeroFloat + 1;

	oneDouble = (double*)(minusOneFloat + 1);
	zeroDouble = oneDouble + 1;
	minusOneDouble = zeroDouble + 1;

	float numberF[3] = { 1, 0, -1 };
	double numberD[3] = { 1, 0, -1 };

	memoryPool->copyFromCPUToGPU(&numberF[0], oneFloat, sizeof(float) * 3);
	memoryPool->copyFromCPUToGPU(&numberD[0], oneDouble, sizeof(double) * 3);
#endif

}

GPUDevice::~GPUDevice() {
	memoryPool->free(oneFloat);

	delete memoryPool;

	CUBLAS_CHECK(cublasDestroy(cublasHandle));
	CURAND_CHECK(curandDestroyGenerator(curandGenerator));

#ifdef HAVE_CUDNN
	CUDNN_CHECK(cudnnDestroy(cudnnHandle));
#endif
}

void* GPUDevice::malloc(size_t size) {
	return memoryPool->malloc(size);
}

void GPUDevice::free(void *ptr) {
	memoryPool->free(ptr);
}

void* GPUDevice::mallocCPU(size_t size) {
	return memoryPool->mallocCPU(size);
}

void GPUDevice::freeCPU(void *ptr) {
	memoryPool->freeCPU(ptr);
}

void GPUDevice::zero(void *ptr, size_t size) {
	memoryPool->zero(ptr, size);
}

void GPUDevice::copy(const void *from, void *to, size_t size) {
	memoryPool->copy(from, to, size);
}

void GPUDevice::copyFromCPUToGPU(const void *from, void *to, size_t size) {
	memoryPool->copyFromCPUToGPU(from, to, size);
}

void GPUDevice::copyFromGPUToCPU(const void *from, void *to, size_t size) {
	memoryPool->copyFromGPUToCPU(from, to, size);
}

void GPUDevice::copyFromGPUToGPU(const void *from, void *to, size_t size) {
	memoryPool->copyFromGPUToGPU(from, to, size);
}

void* GPUDevice::gpuOneHalf() {
#ifdef HAVE_HALF
	return oneHalf;
#else
	DEEP8_RUNTIME_ERROR("not support half");
#endif
}

void* GPUDevice::gpuOneFloat() {
	return oneFloat;
}

void* GPUDevice::gpuOneDouble() {
	return oneDouble;
}

#endif

}