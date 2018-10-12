#include "TensorStorage.h"

namespace Deep8 {

TensorStorage::TensorStorage() : ptr(nullptr), refPtr(nullptr), size(0), device(nullptr) {
}

TensorStorage::TensorStorage(void *p, size_t *refP, size_t s, Device *d) : ptr(p), refPtr(refP), size(s), device(d) {
	DEEP8_ARGUMENT_CHECK(nullptr != p && nullptr != refP && s > 0 && nullptr != device, "the memory is error");

	(*refPtr) = 1;
}

TensorStorage::TensorStorage(const TensorStorage &other) {
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

TensorStorage& TensorStorage::operator=(const TensorStorage &other) {
	if (nullptr != other.refPtr) {
		(*other.refPtr)++;
	}

	if (nullptr != refPtr) {
		(*refPtr)--;

		if (0 == (*refPtr)) {
			free();
		}
	}

	ptr = other.ptr;
	refPtr = other.refPtr;
	size = other.size;
	device = other.device;

	return *this;
}

void TensorStorage::free() {
	if (DeviceType::CPU == device->type) {
		device->free(ptr);
		device->free(refPtr);

		ptr = nullptr;
		refPtr = nullptr;
		size = 0;

		device = nullptr;
	} else {
#ifdef HAVE_CUDA
		device->free(ptr);
		device->freeCPU(refPtr);

		ptr = nullptr;
		refPtr = nullptr;
		size = 0;

		device = nullptr;
#else
		DEEP8_RUNTIME_ERROR("can not call GPU function withou a GPU");
#endif
	}
}

}