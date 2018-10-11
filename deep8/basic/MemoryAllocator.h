#ifndef DEEP8_MEMORYALLOCATOR_H
#define DEEP8_MEMORYALLOCATOR_H

#include "Basic.h"
#include "Exception.h"

namespace Deep8 {

#define DEFAULT_ALIGN 32

/**
 * @brief a Memory allocator class, that it will allocate the memory from the Device
 * for now, only the CPU MemoryAllocator
 */
class MemoryAllocator {
protected:
    size_t align;

public:
    explicit MemoryAllocator(): MemoryAllocator(DEFAULT_ALIGN) {
    }

    explicit MemoryAllocator(size_t align): align(align) {
    }

    virtual ~MemoryAllocator() = default;

    size_t roundUpAlign(size_t n) {
        if (align < 2) {
            return n;
        }

        return ((n + align - 1) / align) * align;
    }

    /**
     * @brief a malloc function that it will get memory from the device
     * @param size the bytes counts
     * @return the pointer that point to memory
     */
    virtual void *malloc(size_t size) = 0;

    /**
     * @brief free the memory
     * @param ptr the pointer that point to the memory
     */
    virtual void free(void *ptr) = 0;

    /**
     * @brief set the memory to zero
     * @param ptr the pointer that point to the memory
     * @param size the bytes count of the memory
     */
    virtual void zero(void *ptr, size_t size) = 0;

    /**
     * copy size memory
     */
    virtual void copy(const void *from, void *to, size_t size) = 0;

	/**
	 * copy memory from host to GPU
	 */
	virtual void copyFromCPUToGPU(const void *from, void *to, size_t size) = 0;

	/**
	 * copy memory from GPU to Host
	 */
	virtual void copyFromGPUToCPU(const void *from, void *to, size_t size) = 0;
	virtual void copyFromGPUToGPU(const void *from, void *to, size_t size) = 0;
};

/**
 * @brief a simple CPU memory allocator
 */
class CPUMemoryAllocator: public MemoryAllocator {
public:
    explicit CPUMemoryAllocator(): CPUMemoryAllocator(DEFAULT_ALIGN) {
    }

    explicit CPUMemoryAllocator(size_t align) : MemoryAllocator(align) {
    }

	void *malloc(size_t size) override {
		void *ptr = _mm_malloc(size, align);

		DEEP8_ASSERT(nullptr != ptr, "system allocate memory error! size is:" << size);

		return ptr;
	}

	void free(void *ptr) override {
		_mm_free(ptr);
	}

	void zero(void *ptr, size_t size) override {
		memset(ptr, 0, size);
	}

	void copy(const void *from, void *to, size_t size) override {
		memcpy(to, from, size);
	}

	/**
	 * copy memory from host to GPU
	 */
	void copyFromCPUToGPU(const void *from, void *to, size_t size) override {
		DEEP8_RUNTIME_ERROR("can not call this function from GPUMemoryAllocator");
	}

	/**
	 * copy memory from GPU to Host
	 */
	void copyFromGPUToCPU(const void *from, void *to, size_t size) override {
		DEEP8_RUNTIME_ERROR("can not call this function from GPUMemoryAllocator");
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) override {
		DEEP8_RUNTIME_ERROR("can not call this function from GPUMemoryAllocator");
	}
};

}

#endif //DEEP8_MEMORYALLOCATOR_H
