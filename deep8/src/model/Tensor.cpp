#include "Tensor.h"

namespace Deep8 {

Tensor::Tensor(): storage(), offset(0), shape(), type(ElementType::from<UnKnownType>()) {
}

Tensor::Tensor(Shape &s): storage(), offset(0), shape(s), type(ElementType::from<UnKnownType>()) {
}

Tensor::Tensor(TensorStorage &ts, size_t off, Shape &s, ElementType et): storage(ts), offset(off), shape(s), type(et) {
}

Tensor::Tensor(TensorStorage &ts, size_t off, std::vector<size_t> &list, ElementType et): Tensor(ts, off, 1, list, et) {
}

Tensor::Tensor(TensorStorage &ts, size_t off, size_t batch, std::vector<size_t> &list, ElementType et): storage(ts), offset(off), shape(batch, list), type(et) {
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
	storage.device->zero(this->raw(), type.byteWidth * shape.size());
}

void Tensor::one() {
	DEEP8_ARGUMENT_CHECK(this->isScalar(), "the tensor is not a scalar");

	if (DeviceType::CPU == this->deviceType()) {
		switch (this->type.id) {
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
		switch (this->type.id) {
		case DType::Float32:
			auto device = this->device();
			device->copyFromGPUToGPU(device->gpuOneFloat(), this->raw(), this->type.byteWidth);
			break;
		case DType::Float64:
			auto device = this->device();
			device->copyFromGPUToGPU(device->gpuOneDouble(), this->raw(), this->type.byteWidth);
			break;
#ifdef HAVE_HALF
		case DType::Float16:
			auto device = this->device();
			device->copyFromGPUToGPU(device->gpuOneHalf(), this->raw(), this->type.byteWidth);
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
	return this->type.byteWidth * this->size();
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

//template<typename T>
//T Tensor<T>::scalar() {
//	DEEP8_ARGUMENT_CHECK(this->isScalar(), "the tensor must be a scalar");
//
//	if (DeviceType::CPU == device()->type) {
//		return data()[0];
//	} else {
//#ifdef HAVE_CUDA
//		T scalar;
//
//		device()->copyFromGPUToCPU(raw(), &scalar, sizeof(T));
//
//		return scalar;
//#else
//		DEEP8_RUNTIME_ERROR("can not call GPU function without a GPU");
//#endif
//	}
//}

//template <typename T>
//std::string Tensor<T>::toString() {
//	std::stringstream ss;
//	ss << "Device Type:" << (DeviceType::CPU == deviceType() ? "CPU" : "GPU");
//	ss << ", Data Type:" << typeStr<T>();
//	ss << ", Data Shape:" << shape.toString();
//	ss << ", ptr:" << raw();
//
//    return ss.str();
//}

///**convert the value to string to print*/
//template <typename T>
//std::string Tensor<T>::valueString() {
//	std::stringstream ss;
//	ss << "[";
//
//	if (DeviceType::CPU == deviceType()) {
//		for (size_t i = 0; i < shape.size(); ++i) {
//			ss << data()[i] << ", ";
//		}
//	} else {
//#ifdef HAVE_CUDA
//		std::vector<T> vec(shape.size());
//
//		device()->copyFromGPUToCPU(raw(), &(vec[0]), sizeof(T) * shape.size());
//
//		for (size_t i = 0; i < shape.size(); ++i) {
//			ss << vec[i] << ", ";
//		}
//#else
//		DEEP8_RUNTIME_ERROR("Can not call GPU function without a GPU");
//#endif
//	}
//
//	ss << "]";
//
//	return ss.str();
//}

}