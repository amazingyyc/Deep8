#include "Tensor.h"

namespace Deep8 {

Tensor::Tensor(): storage(), offset(0), shape(), elementType(ElementType::unknown()) {
}

Tensor::Tensor(Shape &s): storage(), offset(0), shape(s), elementType(ElementType::unknown()) {
}

Tensor::Tensor(TensorStorage &ts, size_t off, Shape &s, ElementType et): storage(ts), offset(off), shape(s), elementType(et) {
}

Tensor::Tensor(TensorStorage &ts, size_t off, std::vector<size_t> &list, ElementType et): Tensor(ts, off, 1, list, et) {
}

Tensor::Tensor(TensorStorage &ts, size_t off, size_t batch, std::vector<size_t> &list, ElementType et): storage(ts), offset(off), shape(batch, list), elementType(et) {
}

DeviceType Tensor::deviceType() {
	return storage.device->type;
}

DeviceType Tensor::deviceType() const {
	return storage.device->type;
}

Device* Tensor::device() {
	return storage.device;
}

Device* Tensor::device() const {
	return storage.device;
}

size_t* Tensor::refPtr() const {
	return storage.refPtr;
}

size_t Tensor::refCount() {
	return storage.refPtr[0];
}

void* Tensor::raw() {
	return (byte*)(storage.ptr) + offset;
}

void* Tensor::raw() const {
	return (byte*)(storage.ptr) + offset;
}

bool Tensor::isScalar() {
	return 1 == shape.size();
}

void Tensor::zero() {
	storage.device->zero(this->raw(), elementType.byteWidth * shape.size());
}

void Tensor::one() {
	DEEP8_ARGUMENT_CHECK(this->isScalar(), "the tensor is not a scalar");

	if (DeviceType::CPU == this->deviceType()) {
		switch (this->elementType.id) {
		case DType::Float32:
			this->data<float>()[0] = 1;
			break;
		case DType::Float64:
			this->data<double>()[0] = 1;
			break;
		default:
			DEEP8_RUNTIME_ERROR("the type is not support");
			break;
		}
	} else {
#ifdef HAVE_CUDA
        Device* device = this->device();

		switch (this->elementType.id) {
		case DType::Float32:
			device->copyFromGPUToGPU(device->gpuOneFloat(), this->raw(), this->elementType.byteWidth);
			break;
		case DType::Float64:
			device->copyFromGPUToGPU(device->gpuOneDouble(), this->raw(), this->elementType.byteWidth);
			break;
#ifdef HAVE_HALF
		case DType::Float16:
			device->copyFromGPUToGPU(device->gpuOneHalf(), this->raw(), this->elementType.byteWidth);
			break;
#endif
		default:
			DEEP8_RUNTIME_ERROR("the type is not support");
			break;
		}
#else
		DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
	}
}

size_t Tensor::byteCount() const  {
	return this->elementType.byteWidth * this->size();
}


size_t Tensor::batch() const {
    return shape.batch;
}

size_t Tensor::nDims() const {
	return shape.nDims;
}

size_t Tensor::size() const {
	return shape.size();
}

size_t Tensor::batchSize() const {
	return shape.batchSize();
}

size_t Tensor::row() const {
	return shape.row();
}

size_t Tensor::col() const {
	return shape.col();
}

size_t Tensor::dim(size_t d) const {
	return shape.dim(d);
}

size_t Tensor::stride(size_t d) const {
	return shape.stride(d);
}

/**release the storage*/
void Tensor::release() {
	storage.release();
}

std::string Tensor::valueStr() {
	std::stringstream ss;
	ss << "[";

	auto size = shape.size();

	if (DeviceType::CPU == deviceType()) {
		if (DType::Float32 == elementType.id) {
			for (size_t i = 0; i < size; ++i) {
				ss << data<float>()[i] << ", ";
			}
		} else if (DType::Float64 == elementType.id) {
			for (size_t i = 0; i < size; ++i) {
				ss << data<double>()[i] << ", ";
			}
		} else {
			DEEP8_RUNTIME_ERROR("the type is error");
		}
	} else {
#ifdef HAVE_CUDA
		if (DType::Float32 == elementType.id) {
			

			std::vector<float> vec(size);

			device()->copyFromGPUToCPU(raw(), &(vec[0]), sizeof(float) * size);

			for (size_t i = 0; i < size; ++i) {
				ss << vec[i] << ", ";
			}
		} else if (DType::Float64 == elementType.id) {
			std::vector<double> vec(size);

			device()->copyFromGPUToCPU(raw(), &(vec[0]), sizeof(double) * size);

			for (size_t i = 0; i < size; ++i) {
				ss << vec[i] << ", ";
			}
#ifdef HAVE_HALF
		} else if (DType::Float16 == elementType.id) {
			std::vector<half> vec(size);

			device()->copyFromGPUToCPU(raw(), &(vec[0]), sizeof(half) * size);

			for (size_t i = 0; i < size; ++i) {
				ss << vec[i] << ", ";
			}
#endif
		} else {
			DEEP8_RUNTIME_ERROR("the type is error");
		}
#else
		DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
	}

	ss << "]";

	return ss.str();
}


}