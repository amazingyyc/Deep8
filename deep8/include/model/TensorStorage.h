#ifndef DEEP8_TENSORSTORAGE_H
#define DEEP8_TENSORSTORAGE_H

#include "Device.h"

namespace Deep8 {

/**
 * the TensorStorage is to store the memory of the Tensor
 * it use the ref count to manage the memory automatic
 */

class TensorStorage {
public:
    /**the device*/
    Device *device;

    /**point to the memory it can the CPU or GPU memory*/
    void *ptr;

    /**store the count that point to this memory, must be a CPU point*/
    size_t *refPtr;

    /**the memory size*/
    size_t size;

	explicit TensorStorage();
	explicit TensorStorage(void *p, size_t *refP, size_t s, Device *d);
	explicit TensorStorage(const TensorStorage &other);

	~TensorStorage();

	TensorStorage& operator=(const TensorStorage &other);

	/**release the reference to current memory*/
	void release();

protected:
	void free();
};

}

#endif //DEEP8_TENSORSTORAGE_H
