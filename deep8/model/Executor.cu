#include "GPUDevice.h"
#include "Executor.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename T>
void Executor<T>::initDeviceGPU() {
	device = new GPUDevice();
}

template <typename T>
Tensor<T> Executor<T>::createTensorWithShapeGPU(Shape &shape) {
	size_t size = sizeof(T) * shape.size();

	auto ptr = device->malloc(size);
	auto refPtr = (size_t*)device->mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, size, device);

	return Tensor<T>(storage, 0, shape);
}

template void Executor<float>::initDeviceGPU();
template void Executor<double>::initDeviceGPU();
#ifdef HAVE_HALF
template void Executor<half>::initDeviceGPU();
#endif

#endif
}