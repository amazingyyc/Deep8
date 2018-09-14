#ifndef DEEP8_DEVICE_H
#define DEEP8_DEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif

#endif

#include "Utils.h"
#include "Exception.h"
#include "CudaException.h"
#include "MemoryAllocator.h"
#include "CPUMemoryPool.h"

namespace Deep8 {

/**
 * @brief the type of the device
 * for now only the CPU is supported
 */
enum class DeviceType {
    CPU,
    GPU,
};

class Device {
public:
    DeviceType type;

    explicit Device(DeviceType type): type(type) {
    }

    virtual ~Device() = default;

    virtual void* malloc(size_t size) = 0;
    virtual void free(void *ptr) = 0;
    virtual void zero(void *ptr, size_t size) = 0;
    virtual void copy(const void *from, void *to, size_t size) = 0;
};

class CPUDevice: public Device {
public:
    CPUMemoryPool *memoryPool;

    /**the Eigen device for using Eigen Tensor*/
    Eigen::NonBlockingThreadPool *eigenThreadPool;
    Eigen::ThreadPoolDevice *eigenDevice;

public:
    explicit CPUDevice(): Device(DeviceType::CPU) {
        memoryPool = new CPUMemoryPool();

        auto threadNum  = getDeviceThreadNum();
        eigenThreadPool = new Eigen::NonBlockingThreadPool(threadNum);
        eigenDevice     = new Eigen::ThreadPoolDevice(eigenThreadPool, threadNum);
    }

    ~CPUDevice() {
        delete memoryPool;
        delete eigenDevice;
        delete eigenThreadPool;
    }

    void* malloc(size_t size) override {
        return memoryPool->malloc(size);
    }

    void free(void *ptr) override {
        memoryPool->free(ptr);
    }

    void zero(void *ptr, size_t size) override {
		memoryPool->zero(ptr, size);
    }

    void copy(const void *from, void *to, size_t size) override {
		memoryPool->copy(from, to, size);
    }
};

#ifdef HAVE_CUDA

class GPUDevice : public Device {
public:
	/**the GPU memory allocator*/
	GPUMemoryAllocator *gpuMemoryAllocator;

	/**the GPU device id*/
	int deviceId;

	/**cuBlas handle*/
	cublasHandle_t cublasHandle;

	/**the GPU memroy contains 1, 0, -1*/
	float *floatOne;
	float *floatZero;
	float *floatMinusOne;

	double *doubleOne;
	double *doubleZero;
	double *doubleMinusOne;

#ifdef HAVE_CUDNN

	/**cudnn handle*/
	cudnnHandle_t cudnnHandle;

#endif

	explicit GPUDevice() : GPUDevice(0) {
	}

	explicit GPUDevice(int deviceId) : Device(DeviceType::GPU), deviceId(deviceId) {
		CUDA_CHECK(cudaSetDevice(deviceId));

		gpuMemoryAllocator = new GPUMemoryAllocator(deviceId);
		
		CUBLAS_CHECK(cublasCreate(&cublasHandle));

#ifdef HAVE_CUDNN
		CUDNN_CHECK(cudnnCreate(&cudnnHandle));
#endif

		void *ptr = gpuMemoryAllocator->malloc(sizeof(float) * 3 + sizeof(double) * 3);

		floatOne = (float*) ptr;
		floatZero = floatOne + 1;
		floatMinusOne = floatZero + 1;

		doubleOne = (double*)(floatMinusOne + 1);
		doubleZero = doubleOne + 1;
		doubleMinusOne = doubleZero + 1;

		float numberF[3] = { 1, 0, -1 };
		float numberD[3] = { 1, 0, -1 };

		gpuMemoryAllocator->copyFromCPUToGPU(&numberF[0], floatOne, sizeof(float) * 3);
		gpuMemoryAllocator->copyFromCPUToGPU(&numberD[0], doubleOne, sizeof(double) * 3);
	}

	~GPUDevice() {
		gpuMemoryAllocator->free(floatOne);

		delete gpuMemoryAllocator;
		cublasDestroy(cublasHandle);

#ifdef HAVE_CUDNN
		cudnnDestroy(cudnnHandle);
#endif
	}

	void* malloc(size_t size) override {
		return gpuMemoryAllocator->malloc(size);
	}

	void free(void *ptr) override {
		gpuMemoryAllocator->free(ptr);
	}

	void zero(void *ptr, size_t size) override {
		gpuMemoryAllocator->zero(ptr, size);
	}

	void copy(const void *from, void *to, size_t size) override {
		gpuMemoryAllocator->copy(from, to, size);
	}

	void copyFromCPUToGPU(const void *from, void *to, size_t size) {
		gpuMemoryAllocator->copyFromCPUToGPU(from, to, size);
	}

	void copyFromGPUToCPU(const void *from, void *to, size_t size) {
		gpuMemoryAllocator->copyFromGPUToCPU(from, to, size);
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) {
		gpuMemoryAllocator->copyFromGPUToGPU(from, to, size);
	}

	template<typename T>
	void *gpuOne() {
		DEEP8_RUNTIME_ERROR("get GPU number error");
	}

	template <>
	void *gpuOne<float>() {
		return floatOne;
	}

	template <>
	void *gpuOne<double>() {
		return doubleOne;
	}

	template<typename T>
	void *gpuZero() {
		DEEP8_RUNTIME_ERROR("get GPU number error");
	}

	template <>
	void *gpuZero<float>() {
		return floatZero;
	}

	template <>
	void *gpuZero<double>() {
		return doubleZero;
	}

	template<typename T>
	void *gpuMinusOne() {
		DEEP8_RUNTIME_ERROR("get GPU number error");
	}

	template <>
	void *gpuMinusOne<float>() {
		return floatMinusOne;
	}

	template <>
	void *gpuMinusOne<double>() {
		return doubleMinusOne;
	}

};

#endif

}

#endif //MAIN_DEVICE_H
