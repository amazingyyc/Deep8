#include "MemoryAllocator.h"

namespace Deep8 {

CPUMemoryAllocator::CPUMemoryAllocator() : CPUMemoryAllocator(DEFAULT_ALIGN) {
}

CPUMemoryAllocator::CPUMemoryAllocator(size_t align) : MemoryAllocator(align) {
}

void* CPUMemoryAllocator::malloc(size_t size) {
	void *ptr = _mm_malloc(size, align);

	DEEP8_ASSERT(nullptr != ptr, "system allocate memory error! size is:" << size);

	return ptr;
}

void CPUMemoryAllocator::free(void *ptr) {
	_mm_free(ptr);
}

void CPUMemoryAllocator::zero(void *ptr, size_t size) {
	memset(ptr, 0, size);
}

void CPUMemoryAllocator::copy(const void *from, void *to, size_t size) {
	memcpy(to, from, size);
}

/**
 * copy memory from host to GPU
 */
void CPUMemoryAllocator::copyFromCPUToGPU(const void *from, void *to, size_t size) {
	DEEP8_RUNTIME_ERROR("can not call this function from GPUMemoryAllocator");
}

/**
 * copy memory from GPU to Host
 */
void CPUMemoryAllocator::copyFromGPUToCPU(const void *from, void *to, size_t size) {
	DEEP8_RUNTIME_ERROR("can not call this function from GPUMemoryAllocator");
}

void CPUMemoryAllocator::copyFromGPUToGPU(const void *from, void *to, size_t size) {
	DEEP8_RUNTIME_ERROR("can not call this function from GPUMemoryAllocator");
}

}