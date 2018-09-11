#ifndef DEEP8_MEMORYALLOCATOR_H
#define DEEP8_MEMORYALLOCATOR_H

#include <cstddef>

#ifdef __GUNC__
#include <mm_malloc.h>
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "Exception.h"
#include "CudaException.h"

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
	void copyToGPU(const void *from, void *to, size_t size) {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
	}

	/**
	 * copy memory from GPU to Host
	 */
	void copyToCPU(const void *from, void *to, size_t size) {
		CUDA_CHECK(cudaSetDevice(deviceId));
		CUDA_CHECK(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
	}
};


#endif


}

#endif //DEEP8_MEMORYALLOCATOR_H
