#include "GPUMemoryAllocator.h"

namespace Deep8 {

#ifdef HAVE_CUDA

GPUMemoryAllocator::GPUMemoryAllocator(int deviceId) : deviceId(deviceId) {
}

void* GPUMemoryAllocator::malloc(size_t size) {
	void *ptr;

	CUDA_CHECK(cudaSetDevice(deviceId));
	CUDA_CHECK(cudaMalloc(&ptr, size));

	return ptr;
}

void GPUMemoryAllocator::free(void *ptr) {
	CUDA_CHECK(cudaSetDevice(deviceId));
	CUDA_CHECK(cudaFree(ptr));
}

void GPUMemoryAllocator::zero(void *ptr, size_t size) {
	CUDA_CHECK(cudaSetDevice(deviceId));
	CUDA_CHECK(cudaMemset(ptr, 0, size));
}

/**
 * for GPU the copy function is between the GPU Device
 */
void GPUMemoryAllocator::copy(const void *from, void *to, size_t size) {
	CUDA_CHECK(cudaSetDevice(deviceId));
	CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice));
}

/**
 * copy memory from host to GPU
 */
void GPUMemoryAllocator::copyFromCPUToGPU(const void *from, void *to, size_t size) {
	CUDA_CHECK(cudaSetDevice(deviceId));
	CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
}

/**
 * copy memory from GPU to Host
 */
void GPUMemoryAllocator::copyFromGPUToCPU(const void *from, void *to, size_t size) {
	CUDA_CHECK(cudaSetDevice(deviceId));
	CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
}

void GPUMemoryAllocator::copyFromGPUToGPU(const void *from, void *to, size_t size) {
	CUDA_CHECK(cudaSetDevice(deviceId));
	CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice));
}

#endif

}