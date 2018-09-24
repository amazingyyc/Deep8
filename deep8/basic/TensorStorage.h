#ifndef DEEP8_TENSORSTORAGE_H
#define DEEP8_TENSORSTORAGE_H


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

    explicit TensorStorage(): ptr(nullptr), refPtr(nullptr), size(0), device(nullptr) {
    }

    explicit TensorStorage(void *p, size_t *refP, size_t s, Device *d)
            : ptr(p), refPtr(refP), size(s), device(d) {
        DEEP8_ARGUMENT_CHECK(nullptr != p && nullptr != refP && s > 0 && nullptr != device, "the memory is error");

        (*refPtr) = 1;
    }

    ~TensorStorage() {
        if (nullptr != refPtr) {
            (*refPtr)--;

            if (0 == (*refPtr)) {
                free();
            }
        }
    }

    TensorStorage &operator=(const TensorStorage &other) {
        (*other.refPtr)++;

        if (nullptr != refPtr) {
            (*refPtr)--;

            if (0 == (*refPtr)) {
                free();
            }
        }

        ptr    = other.ptr;
        refPtr = other.refPtr;
        size   = other.size;
        device = other.device;

        return *this;
    }

    TensorStorage &operator=(const TensorStorage other) {
        (*other.refPtr)++;

        if (nullptr != refPtr) {
            (*refPtr)--;

            if (0 == (*refPtr)) {
                free();
            }
        }

        ptr    = other.ptr;
        refPtr = other.refPtr;
        size   = other.size;
        device = other.device;

        return *this;
    }

    TensorStorage clone() {
        if (DeviceType::CPU == device->type) {
            return cloneCPU();
        } else {
#ifdef HAVE_CUDA
            return cloneGPU();
#else
            DEEP8_RUNTIME_ERROR("can not call GPU function without a GPU");
#endif
        }
    }

protected:
    TensorStorage cloneCPU() {
        auto p    = device->malloc(size);
        auto refP = (size_t*) device->malloc(sizeof(size_t));

        return TensorStorage(p, refP, size, device);
    }

#ifdef HAVE_CUDA
    TensorStorage cloneGPU() {
        auto gpuDevice = static_cast<GPUDevice*>(device);

        auto p    = gpuDevice->malloc(size);
        auto refP = (size_t*) gpuDevice->mallocCPU(sizeof(size_t));

        return TensorStorage(p, refP, size, device);
    }
#endif

    void free() {
        if (DeviceType::CPU == device->type) {
            device->free(ptr);
            device->free(refPtr);

            ptr    = nullptr;
            refPtr = nullptr;
            size   = 0;
        } else {
#ifdef HAVE_CUDA
            auto gpuDevice = static_cast<GPUDevice*>(device);

            gpuDevice->free(ptr);
            gpuDevice->freeCPU(refPtr);

            ptr    = nullptr;
            refPtr = nullptr;
            size   = 0;
#else
            DEEP8_RUNTIME_ERROR("can not call GPU function withou a GPU");
#endif
        }

        device = nullptr;
    }
};

}

#endif //DEEP8_TENSORSTORAGE_H
