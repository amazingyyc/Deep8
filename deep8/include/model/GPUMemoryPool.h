#ifndef DEEP8_GPUMEMORYPOOL_H
#define DEEP8_GPUMEMORYPOOL_H

#include "GPUMemoryAllocator.h"
#include "MemoryPool.h"

namespace Deep8 {

#ifdef HAVE_CUDA

/**
 * the GPU memory pool
 */
struct GPUMemoryChunk {
	GPUMemoryChunk *prev;
	GPUMemoryChunk *next;

	/**the offset of ptr*/
	size_t offset;

	/**the memory size*/
	size_t size;

	/**if this chunk have been used*/
	bool used;

	/**the GPU memory*/
	void *ptr;

	explicit GPUMemoryChunk() : prev(nullptr), next(nullptr), offset(0), size(0), used(false), ptr(nullptr) {
	}
};

/**
 * the GPU memory pool
 */
class GPUMemoryPool {
public:
	/**a CPU memory pool*/
	CPUMemoryPool *cpuMemoryPool;

	/**the memory block size*/
	size_t gpuBlockSize;

	/**a GPU allocator*/
	GPUMemoryAllocator *gpuAllocator;

	/**the GPU id*/
	int deviceId;

	/**the level of the memory*/
	int maxLevel;

	std::vector<GPUMemoryChunk> head;
	std::vector<GPUMemoryChunk> tail;

	/**store the GPU ptr and the chunk*/
	std::unordered_map<void*, GPUMemoryChunk*> ptrMaps;

	explicit GPUMemoryPool();
	explicit GPUMemoryPool(int id);
	explicit GPUMemoryPool(int id, size_t gSize);
	explicit GPUMemoryPool(int id, size_t cSize, size_t gSize);

	~GPUMemoryPool();

protected:
	bool isLinkEmpty(int index);

	void insertToLink(GPUMemoryChunk *chunk, int index);

	GPUMemoryChunk *takeFromLink(int index);

	void allocatorNewBlock();

	/**
	 * the chunk is like a Node of a binary-tree
	 * use the address offset and size to decide if the Chunk is left, right or a root node in the tree
	 * -1 left node
	 * 1 right node
	 * 0 root
	 */
	int chunkType(size_t offset, size_t size);

public:
	void *malloc(size_t size);

	void free(void *ptr);

	void zero(void *ptr, size_t size);

	/**
	 * for GPU the copy function is between the GPU Device
	 */
	void copy(const void *from, void *to, size_t size);

	/**
	 * copy memory from host to GPU
	 */
	void copyFromCPUToGPU(const void *from, void *to, size_t size);

	/**
	 * copy memory from GPU to Host
	 */
	void copyFromGPUToCPU(const void *from, void *to, size_t size);

	void copyFromGPUToGPU(const void *from, void *to, size_t size);

	void *mallocCPU(size_t size);

	void freeCPU(void *ptr);

	std::string toString();

	void printInfo();
};

#endif

}

#endif
