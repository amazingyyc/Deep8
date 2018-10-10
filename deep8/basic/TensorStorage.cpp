#include "Exception.h"
#include "TensorStorage.h"

namespace Deep8 {

explicit TensorStorage::TensorStorage() : ptr(nullptr), refPtr(nullptr), size(0), device(nullptr) {
}

explicit TensorStorage::TensorStorage(void *p, size_t *refP, size_t s, Device *d): ptr(p), refPtr(refP), size(s), device(d) {
	DEEP8_ARGUMENT_CHECK(nullptr != p && nullptr != refP && s > 0 && nullptr != device, "the memory is error");

	(*refPtr) = 1;
}

explicit TensorStorage::TensorStorage(const TensorStorage &other) {
	if (nullptr != other.refPtr) {
		(*other.refPtr)++;
	}

	ptr = other.ptr;
	refPtr = other.refPtr;
	size = other.size;
	device = other.device;
}

TensorStorage::~TensorStorage() {
	if (nullptr != refPtr) {
		(*refPtr)--;

		if (0 == (*refPtr)) {
			free();
		}
	}
}

void TensorStorage::free() {
	if (DeviceType::CPU == device->type) {
		freeCPU();
	} else {
		freeGPU();
	}
}

void TensorStorage::freeCPU() {
	device->free(ptr);
	device->free(refPtr);

	ptr    = nullptr;
	refPtr = nullptr;
	size   = 0;
	
	devive = nullptr;
}



}