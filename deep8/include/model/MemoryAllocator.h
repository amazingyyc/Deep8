#ifndef DEEP8_MEMORYALLOCATOR_H
#define DEEP8_MEMORYALLOCATOR_H

#include "basic/Basic.h"
#include "basic/Exception.h"

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
	explicit CPUMemoryAllocator();

	explicit CPUMemoryAllocator(size_t align);

	void *malloc(size_t size) override;

	void free(void *ptr) override;

	void zero(void *ptr, size_t size) override;

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

}

#endif //DEEP8_MEMORYALLOCATOR_H
