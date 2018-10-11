#ifndef DEEP8_GPUMEMORYALLOCATOR_H
#define DEEP8_GPUMEMORYALLOCATOR_H

#include "GPUBasic.h"
#include "GPUException.h"
#include "MemoryAllocator.h"

namespace Deep8 {

#ifdef HAVE_CUDA

/**
 * GPU memory allocator
 */
class GPUMemoryAllocator : public MemoryAllocator {
public:
	int deviceId;

	explicit GPUMemoryAllocator(int deviceId) : deviceId(deviceId) {
	}

	void *malloc(size_t size) override {
		void *ptr;

		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMalloc(&ptr, size));

		return ptr;
	}

	void free(void *ptr) override {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaFree(ptr));
	}

	void zero(void *ptr, size_t size) override {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMemset(ptr, 0, size));
	}

	/**
	 * for GPU the copy function is between the GPU Device
	 */
	void copy(const void *from, void *to, size_t size) override {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice));
	}

	/**
	 * copy memory from host to GPU
	 */
	void copyFromCPUToGPU(const void *from, void *to, size_t size) override {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
	}

	/**
	 * copy memory from GPU to Host
	 */
	void copyFromGPUToCPU(const void *from, void *to, size_t size) override {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) override {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice));
	}
};

#endif

}

#endif