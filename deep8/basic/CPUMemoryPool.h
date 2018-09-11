#ifndef DEEP8_MEMORYPOOL_H
#define DEEP8_MEMORYPOOL_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>

#include "MathUtils.h"
#include "Exception.h"
#include "MemoryAllocator.h"

namespace Deep8 {

/**
 * define the byte type
 */
typedef char byte;

/**
 * every memory block max size is 2^MAX_MEMORY_BLOCK_RANK
 * the default is 512MB
 */
#define MAX_MEMORY_BLOCK_RANK 29

/**
 * the max memory size is 512MB
 */
#define MAX_MEMORY_BLOCK_SIZE (1 << MAX_MEMORY_BLOCK_RANK)

/**the min size of a memory block*/
#define MIN_MEMORY_BLOCK_SIZE 128

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

    explicit CPUMemoryChunk(): prev(nullptr), next(nullptr), size(0), capacity(0), used(false) {
    }
};

/**
 * a block of memory this block include many chunks
 */
class CPUMemoryBlock {
private:
    /**a pointer of the memory*/
    void *pointer;

    /**the size of this memory must be the power of 2*/
    size_t blockSize;

    /**the fake Head and Tail of chunk list*/
    CPUMemoryChunk head;
    CPUMemoryChunk tail;

    /**
     * the ware store the free memory chunk
     * if the size = 2^k
     * than the ware'size is k + 1
     * ware[i] store the free chunks that the chunk size is 2^i
     * */
    std::vector<std::unordered_set<CPUMemoryChunk*>> ware;

public:
    explicit CPUMemoryBlock(void *ptr, size_t s): pointer(ptr), blockSize(s) {
        DEEP8_ARGUMENT_CHECK(isPowerOf2(blockSize) && blockSize >= MIN_MEMORY_BLOCK_SIZE, "the memory blockSize is error");

        ware.resize(logOf2(blockSize) + 1);

        /**build a chunk*/
        auto chunk = static_cast<CPUMemoryChunk*>(ptr);
        chunk->prev = &head;
        chunk->next = &tail;
        chunk->size = blockSize;
        chunk->capacity = blockSize - sizeof(CPUMemoryChunk);
        chunk->used = false;

        head.next = chunk;
        tail.prev = chunk;

        /**put the chunk to the free ware*/
        ware.back().insert(chunk);
    }

    ~CPUMemoryBlock() {
        ware.clear();
    }

    /**
     * malloc size memory from the block
     * if can not size free memory return null
     */
    void* malloc(size_t size) {
        size += sizeof(CPUMemoryChunk);

        /**the malloc size must be power of 2*/
        size = nextPowerOf2(size);

        if (size > blockSize) {
            return nullptr;
        }

        auto lower = logOf2(size);
        auto index = lower;

        while (index < ware.size()) {
            if (!ware[index].empty()) {
                break;
            }

            index++;
        }

        if (index >= ware.size()) {
            return nullptr;
        }

        while (index > lower) {
            auto chunk = *(ware[index].begin());
            ware[index].erase(chunk);

            auto prev = chunk->prev;
            auto next = chunk->next;

            auto chunkSize = chunk->size / 2;

            /**split the chunk*/
            auto left  = chunk;
            auto right = (CPUMemoryChunk*)((byte*)chunk + chunkSize);

            left->prev = prev;
            left->next = right;
            left->size = chunkSize;
            left->capacity = left->size - sizeof(CPUMemoryChunk);
            left->used = false;

            right->prev = left;
            right->next = next;
            right->size = chunkSize;
            right->capacity = right->size - sizeof(CPUMemoryChunk);
            right->used = false;

            prev->next = left;
            next->prev = right;

            index--;

            ware[index].insert(left);
            ware[index].insert(right);
        }

        auto chunk = *(ware[lower].begin());
        ware[lower].erase(chunk);

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

        auto index = logOf2(chunkSize);

        /**add to free*/
        ware[index].insert(chunk);

        while (index + 1 < ware.size()) {
            /**merge the memory*/
            auto type = chunkType((byte*)chunk - (byte*)pointer, chunk->size);

            if (-1 == type) {
                /**left node merge with right node*/
                auto next = chunk->next;

                if (&tail == next || next->used || next->size != chunk->size) {
                    break;
                }

                /**remove from free ware, and merge it*/
                ware[index].erase(chunk);
                ware[index].erase(next);

                chunk->next = next->next;
                next->next->prev = chunk;

                chunk->size = 2 * chunk->size;
                chunk->capacity = chunk->size - sizeof(CPUMemoryChunk);
                chunk->used = false;

                index++;
                ware[index].insert(chunk);
            } else if (1 == type) {
                auto prev = chunk->prev;

                if (&head == prev || prev->used || prev->size != chunk->size) {
                    break;
                }

                ware[index].erase(chunk);
                ware[index].erase(prev);

                prev->next = chunk->next;
                chunk->next->prev = prev;

                prev->size = 2 * prev->size;
                prev->capacity = prev->size - sizeof(CPUMemoryChunk);
                prev->used = false;

                index++;
                ware[index].insert(prev);

                chunk = prev;
            } else {
                break;
            }
        };
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

        if (size == blockSize) {
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

        DEEP8_RUNTIME_ERROR("chunkType meet a error!");
    }

    /**
     * if the block contain the pointer
     */
    bool contain(void *ptr) {
        return (byte*)(ptr) >= (byte*)(pointer) && (byte*)(ptr) < (byte*)(pointer) + blockSize;
    }

    std::string toString() {
        std::ostringstream oss;
        oss << "The memory block pointer is: " << pointer << ", size is: " << blockSize << "\n";

        auto chunk = head.next;

        while (chunk != &tail) {
            oss << "The chunk size is: " << chunk->size << " byte, capacity is: " << chunk->capacity << " byte, ";

            if (chunk->used) {
                oss << "have been used.";
            } else {
                oss << "not used.";
            }

            oss << "\n";

            chunk = chunk->next;
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

    /**store the memory that allocated by MemoryAllocator*/
    std::vector<std::pair<void*, size_t>> memories;

    /**store the memory blocks*/
    std::vector<CPUMemoryBlock*> memoryBlocks;

    explicit CPUMemoryPool(): CPUMemoryPool(MAX_MEMORY_BLOCK_SIZE) {
    }

    CPUMemoryPool(size_t size) {
        blockSize = nextPowerOf2(size);

        DEEP8_ARGUMENT_CHECK(blockSize >= MIN_MEMORY_BLOCK_SIZE, "the Memory size must >= " << MIN_MEMORY_BLOCK_SIZE);

        allocator = new CPUMemoryAllocator();
    }

    ~CPUMemoryPool() {
        for (auto block : memoryBlocks) {
            delete block;
        }

        for (auto item : memories) {
            allocator->free(item.first);
        }

        memories.clear();
        memoryBlocks.clear();

        delete allocator;
    }

    void* malloc(size_t size) {
		DEEP8_ARGUMENT_CHECK(size > 0, "can not malloc 0 memory");
        DEEP8_ARGUMENT_CHECK(size + sizeof(CPUMemoryChunk) <= blockSize, "malloc too much memory");

        for (auto block : memoryBlocks) {
            auto ptr = block->malloc(size);

            if (nullptr != ptr) {
                return ptr;
            }
        }

        auto ptr = allocator->malloc(blockSize);

        memories.emplace_back(std::make_pair(ptr, blockSize));

        auto block = new CPUMemoryBlock(ptr, blockSize);
        memoryBlocks.emplace_back(block);

        return block->malloc(size);
    }

    void free(void *ptr) {
        for (auto block : memoryBlocks) {
            if (block->contain(ptr)) {
                block->free(ptr);
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

    void free() {
        for (auto block : memoryBlocks) {
            delete block;
        }

        memoryBlocks.clear();

        for (auto item : memories) {
            memoryBlocks.emplace_back(new CPUMemoryBlock(item.first, item.second));
        }
    }

    void printInfo() {
        std::cout << "=============================================================\n" << std::endl;
        std::cout << "the Memory Pool have " << memoryBlocks.size() << " blocks." << std::endl;

        for (auto block : memoryBlocks) {
            std::cout << block->toString() << std::endl;
        }

        std::cout << "=============================================================\n" << std::endl;
    }
};

}

#endif //DEEP8_MEMORYPOOL_H
