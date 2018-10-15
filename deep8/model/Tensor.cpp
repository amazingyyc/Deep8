#include "Tensor.h"

namespace Deep8 {

template<typename T>
Tensor<T>::Tensor() : storage(), offset(0), shape() {
}

template<typename T>
Tensor<T>::Tensor(Shape &s) : storage(), offset(0), shape(s) {
}

template<typename T>
Tensor<T>::Tensor(TensorStorage &ts, size_t off, Shape &s) : storage(ts), offset(off), shape(s) {
}

template<typename T>
Tensor<T>::Tensor(TensorStorage &ts, size_t off, std::initializer_list<size_t> list) : storage(ts), offset(off), shape(list) {
}

template<typename T>
DeviceType Tensor<T>::DeviceType() {
	return storage.device->type;
}

template<typename T>
Device* Tensor<T>::device() {
	return storage.device;
}

template<typename T>
Device* Tensor<T>::device() const {
	return storage.device;
}

template<typename T>
size_t* Tensor<T>::refPtr() const {
	return storage.refPtr;
}

template<typename T>
size_t Tensor<T>::refCount() {
	return storage.refPtr[0];
}

template<typename T>
void* Tensor<T>::raw() {
	return (byte*)(storage.ptr) + offset;
}

template<typename T>
void* Tensor<T>::raw() const {
	return (byte*)(storage.ptr) + offset;
}

template<typename T>
T* Tensor<T>::data() {
	return static_cast<T*>(raw());
}

template<typename T>
T* Tensor<T>::data() const {
	return static_cast<T*>(raw());
}

template<typename T>
bool Tensor<T>::isScalar() {
	return 1 == shape.size();
}

template<typename T>
void Tensor<T>::zero() {
	storage.device->zero(this->raw(), sizeof(T) * shape.size());
}

template<typename T>
size_t Tensor<T>::nDims() const {
	return shape.nDims();
}

template<typename T>
size_t Tensor<T>::size() const {
	return shape.size();
}

template<typename T>
size_t Tensor<T>::batchSize() const {
	return shape.batchSize();
}

template<typename T>
size_t Tensor<T>::batch() const {
	return shape.batch();
}

template<typename T>
size_t Tensor<T>::row() const {
	return shape.row();
}

template<typename T>
size_t Tensor<T>::col() const {
	return shape.col();
}

template<typename T>
size_t Tensor<T>::dim(size_t d) const {
	return shape.dim(d);
}

template<typename T>
T Tensor<T>::scalar() {
	DEEP8_ARGUMENT_CHECK(this->isScalar(), "the tensor must be a scalar");

	if (DeviceType::CPU == device()->type) {
		return data()[0];
	} else {
#ifdef HAVE_CUDA
		T scalar;

		device()->copyFromGPUToCPU(raw(), &scalar, sizeof(T));

		return scalar;
#else 
		DEEP8_RUNTIME_ERROR("can not call GPU function withou a GPU");
#endif
	}
}

DEEP8_DECLARATION_INSTANCE(Tensor);

}