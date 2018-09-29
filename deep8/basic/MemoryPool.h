#ifndef DEEP8_MEMORYPOOL_H
#define DEEP8_MEMORYPOOL_H

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

	void zero(void *ptr, size_t size) {
		allocator->zero(ptr, size);
	}

	void copy(const void *from, void *to, size_t size) {
		allocator->copy(from, to, size);
	}

	void printInfo() {
		std::cout << "============================================================" << std::endl;
		std::cout << toString() << std::endl;
		std::cout << "============================================================" << std::endl;
	}
};

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

	explicit GPUMemoryPool(): GPUMemoryPool(0, DEFAULT_CPU_MEMORY_BLOCK_SIZE_FOR_GPU, DEFAULT_MEMORY_BLOCK_SIZE) {
	}

	explicit GPUMemoryPool(int id) : GPUMemoryPool(id, DEFAULT_CPU_MEMORY_BLOCK_SIZE_FOR_GPU, DEFAULT_MEMORY_BLOCK_SIZE) {
	}

	explicit GPUMemoryPool(int id, size_t gSize) : GPUMemoryPool(id, DEFAULT_CPU_MEMORY_BLOCK_SIZE_FOR_GPU, gSize) {
	}

	explicit GPUMemoryPool(int id, size_t cSize, size_t gSize): deviceId(id) {
		gpuBlockSize = nextPowerOf2(gSize);

		DEEP8_ARGUMENT_CHECK(gpuBlockSize >= MIN_MEMORY_BLOCK_SIZE, "the Memory size must >= " << MIN_MEMORY_BLOCK_SIZE);

		cpuMemoryPool = new CPUMemoryPool(cSize);
		gpuAllocator  = new GPUMemoryAllocator(deviceId);

		maxLevel = (int)logOf2(gpuBlockSize);

		head.resize(maxLevel + 1);
		tail.resize(maxLevel + 1);

		for (int i = 0; i <= maxLevel; ++i) {
			head[i].next = &tail[i];
			head[i].prev = nullptr;

			tail[i].prev = &head[i];
			tail[i].next = nullptr;
		}
	}

	~GPUMemoryPool() {
		auto chunk = head[maxLevel].next;

		while (chunk != &tail[maxLevel]) {
			auto temp = chunk->next;

			chunk->prev->next = chunk->next;
			chunk->next->prev = chunk->prev;

			gpuAllocator->free(chunk->ptr);
			cpuMemoryPool->free(chunk);

			chunk = temp;
		}

		delete cpuMemoryPool;
		delete gpuAllocator;
	}

protected:
	bool isLinkEmpty(int index) {
		DEEP8_ARGUMENT_CHECK(index >= 0 && index <= maxLevel, "the index is error");

		return head[index].next == &tail[index];
	}

	void insertToLink(GPUMemoryChunk *chunk, int index) {
		DEEP8_ARGUMENT_CHECK(index >= 0 && index <= maxLevel, "the index is error");

		chunk->next = head[index].next;
		chunk->prev = &head[index];

		head[index].next = chunk;
		chunk->next->prev = chunk;
	}

	GPUMemoryChunk *takeFromLink(int index) {
		DEEP8_ARGUMENT_CHECK(!isLinkEmpty(index), "the index is empty");

		auto ret = head[index].next;

		head[index].next = ret->next;
		head[index].next->prev = &head[index];

		ret->next = nullptr;
		ret->prev = nullptr;

		return ret;
	}

	void allocatorNewBlock() {
		auto chunk = (GPUMemoryChunk*)cpuMemoryPool->malloc(sizeof(GPUMemoryChunk));

		chunk->ptr = gpuAllocator->malloc(gpuBlockSize);

		chunk->prev = nullptr;
		chunk->next = nullptr;
		chunk->offset = 0;
		chunk->size   = gpuBlockSize;
		chunk->used   = false;

		insertToLink(chunk, maxLevel);

		/**store in map*/
		ptrMaps[chunk->ptr] = chunk;
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

		if (size == this->gpuBlockSize) {
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
		size = nextPowerOf2(size);

		DEEP8_ARGUMENT_CHECK(size > 0, "can not malloc 0 memory");
		DEEP8_ARGUMENT_CHECK(size <= gpuBlockSize, "malloc too much memory");

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

			ptrMaps.erase(chunk->ptr);

			auto left  = chunk;
			auto right = (GPUMemoryChunk*)cpuMemoryPool->malloc(sizeof(GPUMemoryChunk));

			left->prev = nullptr;
			left->next = nullptr;
			left->size /= 2;

			right->offset = left->offset + left->size;
			right->size   = left->size;
			right->used   = false;
			right->ptr    = (byte*)(left->ptr) + left->size;

			insertToLink(right, index - 1);
			insertToLink(left, index - 1);

			ptrMaps[left->ptr]  = left;
			ptrMaps[right->ptr] = right;

			index--;
		}

		auto chunk = takeFromLink(lower);
		chunk->used = true;
		
		return chunk->ptr;
	}

	void free(void *ptr) {
		if (ptrMaps.find(ptr) == ptrMaps.end()) {
			DEEP8_RUNTIME_ERROR("free memory error");
		}

		auto chunk = ptrMaps[ptr];

		DEEP8_ARGUMENT_CHECK(chunk->size > 0 && isPowerOf2(chunk->size), "the memory chunk is error");

		chunk->used = false;

		int index = (int)logOf2(chunk->size);

		insertToLink(chunk, index);

		while (index < maxLevel) {
			auto type = chunkType(chunk->offset, chunk->size);

			if (-1 == type) {
				auto rightPtr = (void*)((byte*)chunk->ptr + chunk->size);

				if (ptrMaps.find(rightPtr) == ptrMaps.end()) {
					DEEP8_RUNTIME_ERROR("merge memory error");
				}

				auto right = ptrMaps[rightPtr];

				if (right->used || chunk->size != right->size) {
					break;
				}

				ptrMaps.erase(chunk->ptr);
				ptrMaps.erase(rightPtr);

				chunk->next->prev = chunk->prev;
				chunk->prev->next = chunk->next;

				right->next->prev = right->prev;
				right->prev->next = right->next;

				chunk->size *= 2;
				chunk->used = false;

				cpuMemoryPool->free(right);

				insertToLink(chunk, index + 1);

				ptrMaps[chunk->ptr] = chunk;

				index++;
			} else if (1 == type) {
				auto leftPtr = (void*)((byte*)chunk->ptr - chunk->size);

				if (ptrMaps.find(leftPtr) == ptrMaps.end()) {
					DEEP8_RUNTIME_ERROR("merge memory error");
				}

				auto left = ptrMaps[leftPtr];

				if (left->used || chunk->size != left->size) {
					break;
				}

				ptrMaps.erase(chunk->ptr);
				ptrMaps.erase(leftPtr);

				chunk->next->prev = chunk->prev;
				chunk->prev->next = chunk->next;

				left->next->prev = left->prev;
				left->prev->next = left->next;

				left->used = false;
				left->size *= 2;

				cpuMemoryPool->free(chunk);

				insertToLink(left, index + 1);

				ptrMaps[left->ptr] = left;

				chunk = left;
				index++;
			} else {
				break;
			}
		}
	}

	void zero(void *ptr, size_t size) {
		gpuAllocator->zero(ptr, size);
	}

	/**
	 * for GPU the copy function is between the GPU Device
	 */
	void copy(const void *from, void *to, size_t size) {
		gpuAllocator->copy(from, to, size);
	}

	/**
	 * copy memory from host to GPU
	 */
	void copyFromCPUToGPU(const void *from, void *to, size_t size) {
		gpuAllocator->copyFromCPUToGPU(from, to, size);
	}

	/**
	 * copy memory from GPU to Host
	 */
	void copyFromGPUToCPU(const void *from, void *to, size_t size) {
		gpuAllocator->copyFromGPUToCPU(from, to, size);
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) {
		gpuAllocator->copyFromGPUToGPU(from, to, size);
	}

	void *mallocCPU(size_t size) {
		return cpuMemoryPool->malloc(size);
	}

	void freeCPU(void *ptr) {
		cpuMemoryPool->free(ptr);
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
		std::cout << "GPU memory info" << std::endl;
		std::cout << toString() << std::endl;
		std::cout << "============================================================" << std::endl;

		std::cout << "============================================================" << std::endl;
		std::cout << "CPU memory info" << std::endl;
		std::cout << cpuMemoryPool->toString() << std::endl;
		std::cout << "============================================================" << std::endl;
	}
};

#endif // HAVE_CUDA

}

#endif //DEEP8_MEMORYPOOL_H
