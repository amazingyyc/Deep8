#ifndef DEEP8_MEMORYALLOCATOR_H
#define DEEP8_MEMORYALLOCATOR_H

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

	void *malloc(size_t size) override;
	void free(void *ptr) override;
	void zero(void *ptr, size_t size) override;
	void copy(const void *from, void *to, size_t size) override;
};


#ifdef HAVE_CUDA

/**
 * GPU memory allocator
 */
class GPUMemoryAllocator : public MemoryAllocator {
public:
	int deviceId;

	explicit GPUMemoryAllocator(int deviceId) : deviceId(deviceId) {
	}

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
	void copyFromCPUToGPU(const void *from, void *to, size_t size);

	/**
	 * copy memory from GPU to Host
	 */
	void copyFromGPUToCPU(const void *from, void *to, size_t size);
	void copyFromGPUToGPU(const void *from, void *to, size_t size);
};


#endif


}

#endif //DEEP8_MEMORYALLOCATOR_H
