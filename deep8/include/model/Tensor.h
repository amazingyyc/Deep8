#ifndef DEEP8_TENSOR_H
#define DEEP8_TENSOR_H

#include "Basic.h"
#include "Exception.h"
#include "Shape.h"
#include "TensorStorage.h"
#include "ElementType.h"

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
        DEEP8_ARGUMENT_CHECK(type.is<T>(), "Tensor type is error, type is: " << type.elementName());

        return (T*) (this->raw());
    }

    template <typename T>
    T* data() const {
        DEEP8_ARGUMENT_CHECK(type.is<T>(), "Tensor type is error, type is: " << type.elementName());

        return (T*)(this->raw());
    }

    template <typename T>
    bool is() {
        return type.is<T>();
    }
};

//class TensorBase {
//public:
//	virtual bool isScalar() = 0;
//	virtual void zero()     = 0;
//
//	virtual size_t nDims() const = 0;
//	virtual size_t size()  const = 0;
//
//	virtual size_t batchSize() const = 0;
//
//	virtual size_t batch() const  = 0;
//	virtual size_t row()   const  = 0;
//	virtual size_t col()   const  = 0;
//
//	virtual size_t dim(size_t d) const  = 0;
//	virtual size_t stride(size_t d) const = 0;
//
//	/**release the storage*/
//	virtual void release() = 0;
//};

///**
// * @brief a Tensor class a multi-dimension array
// * also can represent scalar, vector, matrix
// */
//template <typename T>
//class Tensor: public TensorBase {
//public:
//	/**real store the data*/
//	TensorStorage storage;
//
//	/**
//	 * the memory offset of storage
//	 * the storage size must be >= offset + sizeof(T) * shape.size()
//	 */
//	size_t offset;
//
//	/**the shape of this Tensor*/
//	Shape shape;
//
//public:
//	explicit Tensor();
//	explicit Tensor(Shape&);
//	explicit Tensor(TensorStorage&, size_t off, Shape&);
//	explicit Tensor(TensorStorage&, size_t off, std::vector<size_t>&);
//	explicit Tensor(TensorStorage&, size_t off, size_t batch, std::vector<size_t>&);
//
//public:
//	DeviceType deviceType();
//
//	Device* device();
//	Device* device() const;
//
//	size_t* refPtr() const;
//	size_t refCount();
//
//	void* raw();
//	void* raw() const;
//
//	T* data();
//	T* data() const;
//
//	bool isScalar() override;
//	void zero() override;
//	size_t nDims() const override;
//	size_t size() const override;
//	size_t batchSize() const override;
//	size_t batch() const override;
//	size_t row() const override;
//	size_t col() const override;
//	size_t dim(size_t d) const override;
//	size_t stride(size_t d) const override;
//	
//	/**release the storage*/
//	void release() override;
//
//	T scalar();
//
//};

}

#endif //DEEP8_TENSOR_H
