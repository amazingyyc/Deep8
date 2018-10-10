#include "Basic.h"
#include "Exception.h"
#include "MemoryAllocator.h"

namespace Deep8 {

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

}