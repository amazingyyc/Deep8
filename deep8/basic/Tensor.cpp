#include "Tensor.h"

namespace Deep8 {

template <typename T>
DeviceType Tensor<T>::DeviceType() {
	return storage.device->type;
}

template <typename T>
Device* Tensor<T>::device() {
	return storage.device;
}

template <typename T>
Device* Tensor<T>::device() const {
	return storage.device;
}

template <typename T>
size_t* Tensor<T>::refPtr() const {
	return storage.refPtr;
}

template <typename T>
size_t Tensor<T>::refCount() {
	return storage.refPtr[0];
}

template <typename T>
void* Tensor<T>::raw() {
	return (byte*)(storage.ptr) + offset;
}

template <typename T>
void* Tensor<T>::raw() const {
	return (byte*)(storage.ptr) + offset;
}

template <typename T>
T* Tensor<T>::data() {
	return static_cast<T*>(raw());
}

template <typename T>
T* Tensor<T>::data() const {
	return static_cast<T*>(raw());
}

template <typename T>
bool Tensor<T>::isScalar() {
	return 1 == shape.size();
}

template <typename T>
void Tensor<T>::zero() {
	storage.device->zero(this->raw(), sizeof(T) * shape.size());
}

template <typename T>
size_t Tensor<T>::nDims() const {
	return shape.nDims();
}

template <typename T>
size_t Tensor<T>::size() const {
	return shape.size();
}

template <typename T>
size_t Tensor<T>::batchSize() const {
	return shape.batchSize();
}

template <typename T>
size_t Tensor<T>::batch() const {
	return shape.batch();
}

template <typename T>
size_t Tensor<T>::row() const {
	return shape.row();
}

template <typename T>
size_t Tensor<T>::col() const {
	return shape.col();
}

template <typename T>
size_t Tensor<T>::dim(size_t d) const {
	return shape.dim(d);
}

template <typename T>
T Tensor<T>::scalar() {
	if (DeviceType::CPU == device()->type) {
		return scalarCPU();
	} else {
		return scalarGPU();
	}
}

template <typename T>
T Tensor<T>::scalarCPU() {
	DEEP8_ARGUMENT_CHECK(this->isScalar(), "the tensor must be a scalar");

	return data()[0];
}

template class Tensor<float>;
template class Tensor<double>;

}