#ifndef DEEP8_GPUMEMORYALLOCATOR_H
#define DEEP8_GPUMEMORYALLOCATOR_H

#include "basic/GPUBasic.h"
#include "basic/GPUException.h"
#include "model/MemoryAllocator.h"

namespace Deep8 {

#ifdef HAVE_CUDA

/**
 * GPU memory allocator
 */
class GPUMemoryAllocator : public MemoryAllocator {
public:
	int deviceId;

	explicit GPUMemoryAllocator(int deviceId);

	void *malloc(size_t size) override;

	void free(void *ptr) override;

	void zero(void *ptr, size_t size) override;

	/**
	 * for GPU the copy function is between the GPU Device
	 */
	void copy(const void *from, void *to, size_t size) override;

	/**
	 * copy memory from host to GPU
	 */
	void copyFromCPUToGPU(const void *from, void *to, size_t size) override;

	/**
	 * copy memory from GPU to Host
	 */
	void copyFromGPUToCPU(const void *from, void *to, size_t size) override;

	void copyFromGPUToGPU(const void *from, void *to, size_t size) override;
};

#endif

}

#endif