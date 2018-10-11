#ifndef DEEP8_GPUDEVICE_H
#define DEEP8_GPUDEVICE_H

#include "Device.h"
#include "MemoryPool.h"
#include "GPUMemoryPool.h"

namespace Deep8 {

#ifdef HAVE_CUDA

class GPUDevice : public Device {
public:
	/**the GPU memory allocator*/
	GPUMemoryPool *memoryPool;

	/**the GPU device id*/
	int deviceId;

	/**cuBlas handle*/
	cublasHandle_t cublasHandle;

	/**cuRand generator*/
	curandGenerator_t curandGenerator;

#ifdef HAVE_CUDNN
	/**cudnn handle*/
	cudnnHandle_t cudnnHandle;
#endif

	/**the GPU memroy contains 1, 0, -1*/
	float *oneFloat;
	float *zeroFloat;
	float *minusOneFloat;

	double *oneDouble;
	double *zeroDouble;
	double *minusOneDouble;

#ifdef HAVE_HALF
	half *oneHalf;
	half *zeroHalf;
	half *minusOneHalf;
#endif

	explicit GPUDevice() : GPUDevice(0) {
	}

	explicit GPUDevice(int id) : Device(DeviceType::GPU), deviceId(id) {
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

	~GPUDevice() {
		memoryPool->free(oneFloat);

		delete memoryPool;

		CUBLAS_CHECK(cublasDestroy(cublasHandle));
		CURAND_CHECK(curandDestroyGenerator(curandGenerator));

#ifdef HAVE_CUDNN
		CUDNN_CHECK(cudnnDestroy(cudnnHandle));
#endif
	}

	void* malloc(size_t size) override {
		return memoryPool->malloc(size);
	}

	void free(void *ptr) override {
		memoryPool->free(ptr);
	}

	void *mallocCPU(size_t size) {
		return memoryPool->mallocCPU(size);
	}

	void freeCPU(void *ptr) {
		memoryPool->freeCPU(ptr);
	}

	void zero(void *ptr, size_t size) override {
		memoryPool->zero(ptr, size);
	}

	void copy(const void *from, void *to, size_t size) override {
		memoryPool->copy(from, to, size);
	}

	void copyFromCPUToGPU(const void *from, void *to, size_t size) {
		memoryPool->copyFromCPUToGPU(from, to, size);
	}

	void copyFromGPUToCPU(const void *from, void *to, size_t size) {
		memoryPool->copyFromGPUToCPU(from, to, size);
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) {
		memoryPool->copyFromGPUToGPU(from, to, size);
	}

	void* gpuOneHalf() override {
#ifdef HAVE_HALF
		return oneHalf;
#else
		DEEP8_RUNTIME_ERROR("not support half");
#endif
	}

	void* gpuOneFloat() override {
		return oneFloat;
	}

	void* gpuOneDouble() override {
		return oneDouble;
	}
};

#endif

}

#endif