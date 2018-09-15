#ifndef DEEP8_TENSOR_H
#define DEEP8_TENSOR_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Device.h"
#include "Shape.h"
#include "basic/Exception.h"

namespace Deep8 {

class TensorBase {
public:
	virtual bool isScalar() = 0;
	virtual void zero() = 0;

	virtual size_t nDims() const = 0;
	virtual size_t size() const  = 0;
	virtual size_t batchSize() const  = 0;
	virtual size_t batch() const  = 0;
	virtual size_t row() const  = 0;
	virtual size_t col() const  = 0;
	virtual size_t dim(size_t d) const  = 0;
	virtual void free() = 0;
};

/**
 * @brief a Tensor class a multi-dimension array
 * also can represent scalar, vector, matrix
 */
template <typename T>
class Tensor: public TensorBase {
public:
	/**the pointer of memory*/
	void *pointer;

	/**the shape of this Tensor*/
	Shape shape;

	/**the device of CPU or GPU*/
	Device *device;

public:
	Tensor() : TensorBase() {
	}

	explicit Tensor(void *v, Shape &s, Device *d) : pointer(v), shape(s), device(d) {
	}

	explicit Tensor(void *v, std::initializer_list<size_t> list, Device *d) : pointer(v), shape(list), device(d) {
	}

	void free() override {
		if (nullptr != pointer) {
		    device->free(pointer);
		    pointer = nullptr;
		}
	}

	bool isScalar() override {
		return 1 == shape.size();
	}

	void zero() override {
		device->zero(pointer, sizeof(T) * shape.size());
	}

	size_t nDims() const override{
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

		if (DeviceType::CPU == device->type) {
			return (static_cast<T*>(pointer))[0];
		} else {
#ifdef HAVE_CUDA
			T scalarValue;

			static_cast<GPUDevice*>(device)->copyFromGPUToCPU(pointer, &scalarValue, sizeof(T));

			return scalarValue;
#else
			DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
		}
	}

    T* data() {
        return static_cast<T*>(pointer);
    }

	T* data() const {
		return static_cast<T*>(pointer);
	}
};

typedef Tensor<float> TensorF;

}

#endif //DEEP8_TENSOR_H
