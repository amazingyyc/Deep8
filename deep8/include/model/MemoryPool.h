#ifndef DEEP8_MEMORYPOOL_H
#define DEEP8_MEMORYPOOL_H

#include "utils/MathUtils.h"
#include "model/MemoryAllocator.h"

namespace Deep8 {

/**
 * every memory block max size is 2^MAX_MEMORY_BLOCK_RANK
 * the default is 512MB
 */
#define DEFAULT_MEMORY_BLOCK_RANK 29

 /**
  * the max memory size is 512MB
  */
#define DEFAULT_MEMORY_BLOCK_SIZE (1 << DEFAULT_MEMORY_BLOCK_RANK)

  /**the min size of a memory block*/
#define MIN_MEMORY_BLOCK_SIZE 128

#define DEFAULT_CPU_MEMORY_BLOCK_SIZE_FOR_GPU (1 << 23)

/**
 * a MemoryChunk is a struct store a memory information
 */
struct CPUMemoryChunk {
	/**point to the previous and next memory chunk*/
	CPUMemoryChunk *prev;
	CPUMemoryChunk *next;

	size_t offset;

	/**the size of this chunk include MemoryChunk head*/
	size_t size;

	/**if this chunk have been used*/
	bool used;

	explicit CPUMemoryChunk() : prev(nullptr), next(nullptr), offset(0), size(0), used(false) {
	}
};

class CPUMemoryPool {
public:
	/**memory allocator*/
	CPUMemoryAllocator *allocator;

	/**the block size*/
	size_t blockSize;

	int maxLevel;

	std::vector<CPUMemoryChunk> head;
	std::vector<CPUMemoryChunk> tail;

	explicit CPUMemoryPool();
	explicit CPUMemoryPool(size_t size);

	~CPUMemoryPool();

protected:
	bool isLinkEmpty(int index);

	void insertToLink(CPUMemoryChunk *chunk, int index);

	CPUMemoryChunk *takeFromLink(int index);

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

	std::string toString();

	void printInfo();
};

}

#endif //DEEP8_MEMORYPOOL_H
