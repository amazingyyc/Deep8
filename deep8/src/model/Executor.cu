#include "model/GPUDevice.h"
#include "model/Executor.h"

namespace Deep8 {

void Executor::initDeviceGPU() {
	device = new GPUDevice();
}

Tensor Executor::createTensorGPU(Shape &shape, ElementType type) {
	size_t size = type.byteWidth * shape.size();

	auto ptr = device->malloc(size);
	auto refPtr = (size_t*)device->mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, size, device);

	return Tensor(storage, 0, shape, type);
}

}