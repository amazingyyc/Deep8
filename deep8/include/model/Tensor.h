#ifndef DEEP8_TENSOR_H
#define DEEP8_TENSOR_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/ElementType.h"
#include "model/Shape.h"
#include "model/TensorStorage.h"

namespace Deep8 {

class Tensor {
public:
    /**real store the data*/
    TensorStorage storage;

    /**
     * the memory offset of storage
     * the storage size must be >= offset + sizeof(T) * shape.size()
     */
    size_t offset;

    /**the shape of this Tensor*/
    Shape shape;

    /**the element type*/
    ElementType type;

public:
    explicit Tensor();
    explicit Tensor(Shape&);
    explicit Tensor(TensorStorage&, size_t, Shape&, ElementType);
    explicit Tensor(TensorStorage&, size_t, std::vector<size_t>&, ElementType);
    explicit Tensor(TensorStorage&, size_t, size_t batch, std::vector<size_t>&, ElementType);

public:
    DeviceType deviceType();
    DeviceType deviceType() const;

    Device* device();
    Device* device() const;

    size_t* refPtr() const;
    size_t refCount();

    void* raw();
    void* raw() const;

    bool isScalar();
    void zero();
    void one();

    size_t byteCount() const;

    size_t batch() const;
    size_t nDims() const;

    size_t size()      const;
    size_t batchSize() const;

    size_t row() const;
    size_t col() const;

    size_t dim(size_t d)    const;
    size_t stride(size_t d) const;

    /**release the storage*/
    void release();

    template <typename T>
    T* data() {
        DEEP8_ARGUMENT_CHECK(type.is<T>(), "Tensor type is error, type is: " << type.name);

        return (T*) (this->raw());
    }

    template <typename T>
    T* data() const {
        DEEP8_ARGUMENT_CHECK(type.is<T>(), "Tensor type is error, type is: " << type.name);

        return (T*)(this->raw());
    }

    template <typename T>
    bool is() {
        return type.is<T>();
    }
};

}

#endif //DEEP8_TENSOR_H
