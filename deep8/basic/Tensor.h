#ifndef DEEP8_TENSOR_H
#define DEEP8_TENSOR_H

#include "TensorStorage.h"

namespace Deep8 {

class TensorBase {
public:
	virtual bool isScalar() = 0;
	virtual void zero()     = 0;

	virtual size_t nDims() const = 0;
	virtual size_t size()  const = 0;

	virtual size_t batchSize() const = 0;

	virtual size_t batch() const  = 0;
	virtual size_t row()   const  = 0;
	virtual size_t col()   const  = 0;

	virtual size_t dim(size_t d) const  = 0;
};

/**
 * @brief a Tensor class a multi-dimension array
 * also can represent scalar, vector, matrix
 */
template <typename T>
class Tensor: public TensorBase {
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

public:
	explicit Tensor() : storage(), offset(0), shape() {
	}

	explicit Tensor(Shape &s) : storage(), offset(0), shape(s) {
	}

	explicit Tensor(TensorStorage &ts, size_t off, Shape &s) : storage(ts), offset(off), shape(s) {
	}

	explicit Tensor(TensorStorage &ts, size_t off, std::initializer_list<size_t> list) : storage(ts), offset(off), shape(list) {
	}

	~Tensor() = default;

public:
	DeviceType DeviceType();
	Device* device();

	Device* device() const;


	size_t* refPtr() const;

	size_t refCount();

	void* raw();

	void* raw() const;

	T* data();

	T* data() const;

	bool isScalar() override;

	void zero() override;

	size_t nDims() const override;

	size_t size() const override;

	size_t batchSize() const override;

	size_t batch() const override;

	size_t row() const override;

	size_t col() const override;

	size_t dim(size_t d) const override;

	T scalarCPU();
	T scalarGPU();
	T scalar();
};

}

#endif //DEEP8_TENSOR_H
