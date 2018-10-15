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

	explicit GPUDevice();

	explicit GPUDevice(int id);

	~GPUDevice();

	void* malloc(size_t size) override;

	void free(void *ptr) override;

	void *mallocCPU(size_t size) override;

	void freeCPU(void *ptr) override;

	void zero(void *ptr, size_t size) override;

	void copy(const void *from, void *to, size_t size) override;

	void copyFromCPUToGPU(const void *from, void *to, size_t size) override;

	void copyFromGPUToCPU(const void *from, void *to, size_t size) override;

	void copyFromGPUToGPU(const void *from, void *to, size_t size) override;

	void* gpuOneHalf() override;

	void* gpuOneFloat() override;

	void* gpuOneDouble() override;
};

#endif

}

#endif