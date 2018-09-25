#ifndef DEEP8_MEMORYPOOL_H
#define DEEP8_MEMORYPOOL_H

namespace Deep8 {

/**
 * define the byte type
 */
typedef char byte;

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

	/**the size of this chunk include MemoryChunk head*/
	size_t size;

	/**the capacity of this chunk not include the Chunk head*/
	size_t capacity;

	/**if this chunk have been used*/
	bool used;

	explicit CPUMemoryChunk() : prev(nullptr), next(nullptr), size(0), capacity(0), used(false) {
	}
};

class CPUMemoryPool;

class CPUMemoryBlock {
	friend CPUMemoryPool;

private:
	/**the pointer of memory block*/
	void *ptr;

	/**the size of this memory block*/
	size_t size;

	/**
	 * the fake head and tail of the double-list
	 * head[i] store the free chunks that the chunk size is 2^i
	 */
	std::vector<CPUMemoryChunk> head;
	std::vector<CPUMemoryChunk> tail;

public:
	explicit CPUMemoryBlock(void *p, size_t s) : ptr(p), size(s) {
		DEEP8_ARGUMENT_CHECK(isPowerOf2(size) && size >= MIN_MEMORY_BLOCK_SIZE, "the memory blockSize is error");

		int count = (int)logOf2(size) + 1;

		head.resize(count);
		tail.resize(count);

		for (int i = 0; i < count; ++i) {
			head[i].next = &tail[i];
			head[i].prev = nullptr;
			head[i].used = true;

			tail[i].prev = &head[i];
			tail[i].next = nullptr;
			tail[i].used = true;
		}

		insert(ptr, count - 1);
	}

protected:
	void insertChunk(CPUMemoryChunk *chunk, int index) {
		chunk->next = head[index].next;
		chunk->prev = &head[index];

		head[index].next = chunk;
		chunk->next->prev = chunk;
	}

	void insert(void *ptr, int index) {
		DEEP8_ARGUMENT_CHECK(index >= 0 && index < head.size(), "the index is error");

		auto chunk = static_cast<CPUMemoryChunk*>(ptr);
		chunk->used = false;
		chunk->size = (size_t)(1 << index);
		chunk->capacity = chunk->size - sizeof(CPUMemoryChunk);

		chunk->next = head[index].next;
		chunk->prev = &head[index];

		head[index].next = chunk;
		chunk->next->prev = chunk;
	}

	CPUMemoryChunk *take(int index) {
		DEEP8_ARGUMENT_CHECK(!empty(index), "the index is empty");

		auto ret = head[index].next;

		head[index].next = ret->next;
		head[index].next->prev = &head[index];

		ret->next = nullptr;
		ret->prev = nullptr;

		return ret;
	}

	bool empty(int index) {
		DEEP8_ARGUMENT_CHECK(index >= 0 && index < head.size(), "the index is error");

		return head[index].next == &tail[index];
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

		if (size == this->size) {
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
	void *malloc(size_t mem) {
		mem += sizeof(CPUMemoryChunk);

		/**the malloc size must be power of 2*/
		mem = nextPowerOf2(mem);

		if (mem > this->size) {
			return nullptr;
		}

		int lower = (int)logOf2(mem);
		int index = lower;

		while (index < head.size()) {
			if (!empty(index)) {
				break;
			}

			index++;
		}

		if (index >= head.size()) {
			return nullptr;
		}

		while (index > lower) {
			auto chunk = take(index);

			insert((byte*)chunk, index - 1);
			insert((byte*)chunk + (1 << (index - 1)), index - 1);

			index--;
		}

		auto chunk = take(lower);
		chunk->used = true;

		return (byte*)chunk + sizeof(CPUMemoryChunk);
	}

	void free(void *ptr) {
		if (!contain(ptr)) {
			return;
		}

		auto chunk = (CPUMemoryChunk*)((byte*)ptr - sizeof(CPUMemoryChunk));
		chunk->used = false;

		auto chunkSize = chunk->size;

		DEEP8_ARGUMENT_CHECK(chunkSize > 0 && isPowerOf2(chunkSize), "the memory chunk is error");

		int index = (int)logOf2(chunkSize);

		insertChunk(chunk, index);

		while (index + 1 < head.size()) {
			/**merge the memory*/
			auto type = chunkType((byte*)chunk - (byte*)(this->ptr), chunk->size);

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
				chunk->capacity = chunk->size - sizeof(CPUMemoryChunk);

				index++;

				insertChunk(chunk, index);
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
				left->capacity = left->size - sizeof(CPUMemoryChunk);

				index++;

				insertChunk(left, index);

				chunk = left;
			} else {
				break;
			}
		}
	}

	bool contain(void *p) {
		return ((byte*)p >= (byte*)ptr) && ((byte*)p < (byte*)ptr + size);
	}

	std::string toString() {
		std::ostringstream oss;
		oss << "The memory block pointer is: " << ptr << ", size is: " << size << "\n";

		for (int i = 0; i < head.size(); ++i) {
			auto chunk = head[i].next;

			while (chunk != &tail[i]) {
				oss << "The chunk size is: " << chunk->size << " byte, capacity is: " << chunk->capacity << " byte, ";

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
};

class CPUMemoryPool {
public:
	/**the memory block size*/
	size_t blockSize;

	/**for allocate the memory*/
	CPUMemoryAllocator *allocator;

	/**store the memory blocks*/
	std::vector<CPUMemoryBlock*> blocks;

	explicit CPUMemoryPool() : CPUMemoryPool(DEFAULT_MEMORY_BLOCK_SIZE) {
	}

	CPUMemoryPool(size_t size) {
		blockSize = nextPowerOf2(size);

		DEEP8_ARGUMENT_CHECK(blockSize >= MIN_MEMORY_BLOCK_SIZE, "the Memory size must >= " << MIN_MEMORY_BLOCK_SIZE);

		allocator = new CPUMemoryAllocator();
	}

	~CPUMemoryPool() {
		for (auto item : blocks) {
			allocator->free(item->ptr);
			delete item;
		}

		blocks.clear();

		delete allocator;
	}

	void* malloc(size_t size) {
		DEEP8_ARGUMENT_CHECK(size > 0, "can not malloc 0 memory");
		DEEP8_ARGUMENT_CHECK(size + sizeof(CPUMemoryChunk) <= blockSize, "malloc too much memory");

		for (auto item : blocks) {
			auto ptr = item->malloc(size);

			if (nullptr != ptr) {
				return ptr;
			}
		}

		auto ptr = allocator->malloc(blockSize);

		auto item = new CPUMemoryBlock(ptr, blockSize);
		blocks.emplace_back(item);

		return item->malloc(size);
	}

	void free(void *ptr) {
		for (auto item : blocks) {
			if (item->contain(ptr)) {
				item->free(ptr);
				return;
			}
		}

		DEEP8_RUNTIME_ERROR("free memory error");
	}

	void zero(void *ptr, size_t size) {
		allocator->zero(ptr, size);
	}

	void copy(const void *from, void *to, size_t size) {
		allocator->copy(from, to, size);
	}

	void printInfo() {
		std::cout << "=============================================================\n" << std::endl;
		std::cout << "the Memory Pool have " << blocks.size() << " blocks." << std::endl;

		for (auto block : blocks) {
			std::cout << block->toString() << std::endl;
		}

		std::cout << "=============================================================\n" << std::endl;
	}
};


/#ifdef HAVE_CUDA

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

	explicit GPUMemoryPool(int id, size_t cSize, size_t gSize): deviceId(id) {
		gpuBlockSize = nextPowerOf2(gSize);

		DEEP8_ARGUMENT_CHECK(gpuBlockSize >= MIN_MEMORY_BLOCK_SIZE, "the Memory size must >= " << MIN_MEMORY_BLOCK_SIZE);

		cpuMemoryPool = new CPUMemoryPool(cSize);
		gpuAllocator  = new GPUMemoryAllocator(deviceId);

		int maxLevel = (int)logOf2(gpuBlockSize);

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

		while (chunk != tail[maxLevel]) {
			auto temp = chunk->next;

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

		insertToLink(chunk, index);

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

		if (size == this->size) {
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
			right->use    = false;
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
};

#endif // HAVE_CUDA

}

#endif //DEEP8_MEMORYPOOL_H
