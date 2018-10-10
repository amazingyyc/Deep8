#ifndef DEEP8_GPUDEVICE_H
#define DEEP8_GPUDEVICE_H

#include "CudaHeads.h"
#include "Device.h"

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
	float *floatOne;
	float *floatZero;
	float *floatMinusOne;

	double *doubleOne;
	double *doubleZero;
	double *doubleMinusOne;

#ifdef HAVE_HALF
	half *halfOne;
	half *halfZero;
	half *halfMinusOne;
#endif // HAVE_HALF


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

		floatOne = (float*)ptr;
		floatZero = floatOne + 1;
		floatMinusOne = floatZero + 1;

		doubleOne = (double*)(floatMinusOne + 1);
		doubleZero = doubleOne + 1;
		doubleMinusOne = doubleZero + 1;

		halfOne = (half*)(doubleMinusOne + 1);
		halfZero = halfOne + 1;
		halfMinusOne = halfZero + 1;

		float  numberF[3] = { 1, 0, -1 };
		double numberD[3] = { 1, 0, -1 };
		half   numberH[3] = { 1.0, 0.0, -1.0 };

		memoryPool->copyFromCPUToGPU(&numberF[0], floatOne, sizeof(float) * 3);
		memoryPool->copyFromCPUToGPU(&numberD[0], doubleOne, sizeof(double) * 3);
		memoryPool->copyFromCPUToGPU(&numberH[0], halfOne, sizeof(half) * 3);

#else
		void *ptr = memoryPool->malloc(sizeof(float) * 3 + sizeof(double) * 3);

		floatOne = (float*)ptr;
		floatZero = floatOne + 1;
		floatMinusOne = floatZero + 1;

		doubleOne = (double*)(floatMinusOne + 1);
		doubleZero = doubleOne + 1;
		doubleMinusOne = doubleZero + 1;

		float numberF[3] = { 1, 0, -1 };
		double numberD[3] = { 1, 0, -1 };

		memoryPool->copyFromCPUToGPU(&numberF[0], floatOne, sizeof(float) * 3);
		memoryPool->copyFromCPUToGPU(&numberD[0], doubleOne, sizeof(double) * 3);
#endif // HAVE_HALF
	}

	~GPUDevice() {
		memoryPool->free(floatOne);

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

	template<typename T>
	void *gpuOne() {
		DEEP8_RUNTIME_ERROR("get GPU number value error");
	}

	template <>
	void *gpuOne<float>() {
		return floatOne;
	}

	template <>
	void *gpuOne<double>() {
		return doubleOne;
	}

#ifdef HAVE_HALF
	template <>
	void *gpuOne<half>() {
		return halfOne;
	}
#endif // HAVE_HALF


	template<typename T>
	void *gpuZero() {
		DEEP8_RUNTIME_ERROR("get GPU number value error");
	}

	template <>
	void *gpuZero<float>() {
		return floatZero;
	}

	template <>
	void *gpuZero<double>() {
		return doubleZero;
	}

#ifdef HAVE_HALF
	template <>
	void *gpuZero<half>() {
		return halfZero;
	}
#endif // HAVE_HALF

	template<typename T>
	void *gpuMinusOne() {
		DEEP8_RUNTIME_ERROR("get GPU number value error");
	}

	template <>
	void *gpuMinusOne<float>() {
		return floatMinusOne;
	}

	template <>
	void *gpuMinusOne<double>() {
		return doubleMinusOne;
	}

#ifdef HAVE_HALF
	template <>
	void *gpuMinusOne<half>() {
		return halfMinusOne;
	}
#endif // HAVE_HALF

};

#endif

}

#endif