#ifndef DEEP8_TENSOR_H
#define DEEP8_TENSOR_H

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
	DeviceType DeviceType() {
		return storage.device->type;
	}

	Device* device() {
		return storage.device;
	}

	Device* device() const {
		return storage.device;
	}


	size_t* refPtr() const {
		return storage.refPtr;
	}

	size_t refCount() {
		return storage.refPtr[0];
	}

	void* raw() {
		return (byte*)(storage.ptr) + offset;
	}

	void* raw() const {
		return (byte*)(storage.ptr) + offset;
	}

	T* data() {
		return static_cast<T*>(raw());
	}

	T* data() const {
		return static_cast<T*>(raw());
	}

	bool isScalar() override {
		return 1 == shape.size();
	}

	void zero() override {
		storage.device->zero(this->raw(), sizeof(T) * shape.size());
	}

	size_t nDims() const override {
		return shape.nDims();
	}

	size_t size() const override {
		return shape.size();
	}

	size_t batchSize() const override {
		return shape.batchSize();
	}

	size_t batch() const override {
		return shape.batch();
	}

	size_t row() const override {
		return shape.row();
	}

	size_t col() const override {
		return shape.col();
	}

	size_t dim(size_t d) const override {
		return shape.dim(d);
	}

	T scalar() {
		DEEP8_ARGUMENT_CHECK(this->isScalar(), "the tensor must be a scalar");

		if (DeviceType::CPU == device()->type) {
			return data()[0];
		} else {
#ifdef HAVE_CUDA
			T scalar;

			static_cast<GPUDevice*>(device)->copyFromGPUToCPU(raw(), &scalar, sizeof(T));

			return scalar;
#else
			DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
		}
	}
};

}

#endif //DEEP8_TENSOR_H
