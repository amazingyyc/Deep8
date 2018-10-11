#ifndef DEEP8_MEMORYPOOL_H
#define DEEP8_MEMORYPOOL_H

#include "../utils/MathUtils.h"
#include "MemoryAllocator.h"

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

	explicit CPUMemoryPool(): CPUMemoryPool(DEFAULT_MEMORY_BLOCK_SIZE) {
	}

	explicit CPUMemoryPool(size_t size) {
		blockSize = nextPowerOf2(size);

		DEEP8_ARGUMENT_CHECK(blockSize >= MIN_MEMORY_BLOCK_SIZE, "the Memory size must >= " << MIN_MEMORY_BLOCK_SIZE);

		allocator = new CPUMemoryAllocator();

		maxLevel = (int)logOf2(blockSize);

		head.resize(maxLevel + 1);
		tail.resize(maxLevel + 1);

		for (int i = 0; i <= maxLevel; ++i) {
			head[i].next = &tail[i];
			head[i].prev = nullptr;

			tail[i].prev = &head[i];
			tail[i].next = nullptr;
		}
	}

	~CPUMemoryPool() {
		auto chunk = head[maxLevel].next;

		while (chunk != &tail[maxLevel]) {
			auto temp = chunk->next;

			chunk->prev->next = chunk->next;
			chunk->next->prev = chunk->prev;

			allocator->free(chunk);

			chunk = temp;
		}

		delete allocator;
	}

protected:
	bool isLinkEmpty(int index) {
		DEEP8_ARGUMENT_CHECK(index >= 0 && index <= maxLevel, "the index is error");

		return head[index].next == &tail[index];
	}

	void insertToLink(CPUMemoryChunk *chunk, int index) {
		DEEP8_ARGUMENT_CHECK(index >= 0 && index <= maxLevel, "the index is error");

		chunk->next = head[index].next;
		chunk->prev = &head[index];

		head[index].next = chunk;
		chunk->next->prev = chunk;
	}

	CPUMemoryChunk *takeFromLink(int index) {
		DEEP8_ARGUMENT_CHECK(!isLinkEmpty(index), "the index is empty");

		auto ret = head[index].next;

		head[index].next = ret->next;
		head[index].next->prev = &head[index];

		ret->next = nullptr;
		ret->prev = nullptr;

		return ret;
	}

	void allocatorNewBlock() {
		auto chunk = (CPUMemoryChunk*) allocator->malloc(blockSize);

		chunk->prev = nullptr;
		chunk->next = nullptr;
		chunk->offset = 0;
		chunk->size   = blockSize;
		chunk->used   = false;

		insertToLink(chunk, maxLevel);
	}

	/**
	 * the chunk is like a Node of a binary-tree
	 * use the address offset and size to decide if the Chunk is left, right or a root node in the tree
	 * -1 left node
	 * 1 right node
	 * 0 root
	 */
	int chunkType(size_t offset, size_t size) {
		DEEP8_ARGUMENT_CHECK(size > 0, "cannot call function chunkType with size is 0");

		if (size == this->blockSize) {
			return 0;
		}

		while (offset > size) {
			offset -= prevPowerOf2(offset);
		}

		if (offset == size) {
			return 1;
		} else if (0 == offset) {
			return -1;
		}

		DEEP8_RUNTIME_ERROR("chunk type is error!");
	}

public:
	void *malloc(size_t size) {
		size += sizeof(CPUMemoryChunk);
		size = nextPowerOf2(size);

		DEEP8_ARGUMENT_CHECK(size > 0, "can not malloc 0 memory");
		DEEP8_ARGUMENT_CHECK(size <= blockSize, "malloc too much memory");

		int lower = (int)logOf2(size);
		int index = lower;

		while (index <= maxLevel && isLinkEmpty(index)) {
			index++;
		}

		/**no extra memory. should allocator more*/
		if (index > maxLevel) {
			allocatorNewBlock();
			index = maxLevel;
		}

		while (index > lower) {
			auto chunk = takeFromLink(index);

			auto left  = chunk;
			auto right = (CPUMemoryChunk*)((byte*)chunk + chunk->size / 2);

			left->size /= 2;

			right->offset = left->offset + left->size;
			right->size   = left->size;
			right->used   = false;
			right->prev   = nullptr;
			right->next   = nullptr;

			insertToLink(right, index - 1);
			insertToLink(left, index - 1);

			index--;
		}

		auto chunk = takeFromLink(lower);
		chunk->used = true;

		return (byte*)chunk + sizeof(CPUMemoryChunk);
	}

	void free(void *ptr) {
		auto chunk = (CPUMemoryChunk*)((byte*)ptr - sizeof(CPUMemoryChunk));

		DEEP8_ARGUMENT_CHECK(chunk->size > 0 && isPowerOf2(chunk->size), "the memory chunk is error");

		chunk->used = false;

		int index = (int)logOf2(chunk->size);

        insertToLink(chunk, index);

		while (index < maxLevel) {
			auto type = chunkType(chunk->offset, chunk->size);

			if (-1 == type) {
				 auto right = (CPUMemoryChunk*)((byte*)chunk + chunk->size);

				 if (right->used || chunk->size != right->size) {
					break;
				}

				chunk->next->prev = chunk->prev;
				chunk->prev->next = chunk->next;

				right->next->prev = right->prev;
				right->prev->next = right->next;

				chunk->used = false;
				chunk->size = 2 * chunk->size;

                insertToLink(chunk, index + 1);

				index++;
			} else if (1 == type) {
				auto left = (CPUMemoryChunk*)((byte*)chunk - chunk->size);

				if (left->used || chunk->size != left->size) {
					break;
				}

				chunk->next->prev = chunk->prev;
				chunk->prev->next = chunk->next;

				left->next->prev = left->prev;
				left->prev->next = left->next;

				left->used = false;
				left->size = 2 * left->size;

                insertToLink(left, index);

				index++;

				chunk = left;
			} else {
				break;
			}
		}
	}

	void zero(void *ptr, size_t size) {
		allocator->zero(ptr, size);
	}

	void copy(const void *from, void *to, size_t size) {
		allocator->copy(from, to, size);
	}

	/**
	 * copy memory from host to GPU
	 */
	void copyFromCPUToGPU(const void *from, void *to, size_t size) {
		allocator->copyFromCPUToGPU(from, to, size);
	}

	/**
	 * copy memory from GPU to Host
	 */
	void copyFromGPUToCPU(const void *from, void *to, size_t size) {
		allocator->copyFromGPUToCPU(from, to, size);
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) {
		allocator->copyFromGPUToGPU(from, to, size);
	}

	std::string toString() {
		std::ostringstream oss;

		for (int i = 0; i < head.size(); ++i) {
			auto chunk = head[i].next;

			while (chunk != &tail[i]) {
				oss << "The chunk offset is: " << chunk->offset << ", size is: " << chunk->size << " byte, ";

				if (chunk->used) {
					oss << "have been used.";
				} else {
					oss << "not used.";
				}

				oss << "\n";

				chunk = chunk->next;
			}
		}

		return oss.str();
	}

	void printInfo() {
		std::cout << "============================================================" << std::endl;
		std::cout << toString() << std::endl;
		std::cout << "============================================================" << std::endl;
	}
};

}

#endif //DEEP8_MEMORYPOOL_H
