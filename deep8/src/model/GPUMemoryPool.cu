#include "GPUMemoryPool.h"

namespace Deep8 {

#ifdef HAVE_CUDA

GPUMemoryPool::GPUMemoryPool() : GPUMemoryPool(0, DEFAULT_CPU_MEMORY_BLOCK_SIZE_FOR_GPU, DEFAULT_MEMORY_BLOCK_SIZE) {
}

GPUMemoryPool::GPUMemoryPool(int id) : GPUMemoryPool(id, DEFAULT_CPU_MEMORY_BLOCK_SIZE_FOR_GPU, DEFAULT_MEMORY_BLOCK_SIZE) {
}

GPUMemoryPool::GPUMemoryPool(int id, size_t gSize) : GPUMemoryPool(id, DEFAULT_CPU_MEMORY_BLOCK_SIZE_FOR_GPU, gSize) {
}

GPUMemoryPool::GPUMemoryPool(int id, size_t cSize, size_t gSize) : deviceId(id) {
	gpuBlockSize = nextPowerOf2(gSize);

	DEEP8_ARGUMENT_CHECK(gpuBlockSize >= MIN_MEMORY_BLOCK_SIZE, "the Memory size must >= " << MIN_MEMORY_BLOCK_SIZE);

	cpuMemoryPool = new CPUMemoryPool(cSize);
	gpuAllocator = new GPUMemoryAllocator(deviceId);

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

GPUMemoryPool::~GPUMemoryPool() {
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


bool GPUMemoryPool::isLinkEmpty(int index) {
	DEEP8_ARGUMENT_CHECK(index >= 0 && index <= maxLevel, "the index is error");

	return head[index].next == &tail[index];
}

void GPUMemoryPool::insertToLink(GPUMemoryChunk *chunk, int index) {
	DEEP8_ARGUMENT_CHECK(index >= 0 && index <= maxLevel, "the index is error");

	chunk->next = head[index].next;
	chunk->prev = &head[index];

	head[index].next = chunk;
	chunk->next->prev = chunk;
}

GPUMemoryChunk* GPUMemoryPool::takeFromLink(int index) {
	DEEP8_ARGUMENT_CHECK(!isLinkEmpty(index), "the index is empty");

	auto ret = head[index].next;

	head[index].next = ret->next;
	head[index].next->prev = &head[index];

	ret->next = nullptr;
	ret->prev = nullptr;

	return ret;
}

void GPUMemoryPool::allocatorNewBlock() {
	auto chunk = (GPUMemoryChunk*)cpuMemoryPool->malloc(sizeof(GPUMemoryChunk));

	chunk->ptr = gpuAllocator->malloc(gpuBlockSize);

	chunk->prev = nullptr;
	chunk->next = nullptr;
	chunk->offset = 0;
	chunk->size = gpuBlockSize;
	chunk->used = false;

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
int GPUMemoryPool::chunkType(size_t offset, size_t size) {
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

void* GPUMemoryPool::malloc(size_t size) {
    if (0 == size) {
        return nullptr;
    }

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

		auto left = chunk;
		auto right = (GPUMemoryChunk*)cpuMemoryPool->malloc(sizeof(GPUMemoryChunk));

		left->prev = nullptr;
		left->next = nullptr;
		left->size /= 2;

		right->offset = left->offset + left->size;
		right->size = left->size;
		right->used = false;
		right->ptr = (byte*)(left->ptr) + left->size;

		insertToLink(right, index - 1);
		insertToLink(left, index - 1);

		ptrMaps[left->ptr] = left;
		ptrMaps[right->ptr] = right;

		index--;
	}

	auto chunk = takeFromLink(lower);
	chunk->used = true;

	return chunk->ptr;
}

void GPUMemoryPool::free(void *ptr) {
    if (nullptr == ptr) {
        return;
    }

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

void GPUMemoryPool::zero(void *ptr, size_t size) {
	gpuAllocator->zero(ptr, size);
}

/**
	* for GPU the copy function is between the GPU Device
	*/
void GPUMemoryPool::copy(const void *from, void *to, size_t size) {
	gpuAllocator->copy(from, to, size);
}

/**
	* copy memory from host to GPU
	*/
void GPUMemoryPool::copyFromCPUToGPU(const void *from, void *to, size_t size) {
	gpuAllocator->copyFromCPUToGPU(from, to, size);
}

/**
	* copy memory from GPU to Host
	*/
void GPUMemoryPool::copyFromGPUToCPU(const void *from, void *to, size_t size) {
	gpuAllocator->copyFromGPUToCPU(from, to, size);
}

void GPUMemoryPool::copyFromGPUToGPU(const void *from, void *to, size_t size) {
	gpuAllocator->copyFromGPUToGPU(from, to, size);
}

void* GPUMemoryPool::mallocCPU(size_t size) {
	return cpuMemoryPool->malloc(size);
}

void GPUMemoryPool::freeCPU(void *ptr) {
	cpuMemoryPool->free(ptr);
}

std::string GPUMemoryPool::toString() {
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

void GPUMemoryPool::printInfo() {
	std::cout << "============================================================" << std::endl;
	std::cout << "GPU memory info" << std::endl;
	std::cout << toString() << std::endl;
	std::cout << "============================================================" << std::endl;

	std::cout << "============================================================" << std::endl;
	std::cout << "CPU memory info" << std::endl;
	std::cout << cpuMemoryPool->toString() << std::endl;
	std::cout << "============================================================" << std::endl;
}

#endif

}