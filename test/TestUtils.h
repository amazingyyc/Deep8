#ifndef DEEP8_TESTUTILS_H
#define DEEP8_TESTUTILS_H

#include <stdarg.h>

#include "model/ElementType.h"
#include "model/Shape.h"
#include "model/Tensor.h"
#include "nodes/Variable.h"

namespace Deep8 {

Tensor createTensor(CPUDevice &device, ElementType type, size_t batch, std::vector<size_t> list) {
	Shape shape(batch, list);

	auto storageSize = type.byteWidth * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*) device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	if (type.id == DType::Float32) {
		auto generator = [&]() -> float {
			return rand() % 50;
		};

		std::generate((float*)ptr, (float*)ptr + shape.size(), generator);
	} else if (type.id == DType::Float64) {
		auto generator = [&]() -> double {
			return rand() % 50;
		};

		std::generate((double*)ptr, (double*)ptr + shape.size(), generator);
	} else {
		DEEP8_RUNTIME_ERROR("type is error");
	}

	return Tensor(storage, 0, shape, type);
}


#ifdef HAVE_CUDA
Tensor createTensor(GPUDevice& device, ElementType type, size_t batch, std::vector<size_t> list) {
    Shape shape(batch, list);

    auto storageSize = type.byteWidth * shape.size();

    auto ptr = device.malloc(storageSize);
    auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

    TensorStorage storage(ptr, refPtr, storageSize, &device);

    return Tensor(storage, 0, shape, type);
}

Tensor createTensor(GPUDevice& device, void* cpuptr, ElementType type, size_t batch, std::vector<size_t> list) {
    Shape shape(batch, list);

    auto storageSize = type.byteWidth * shape.size();

    auto ptr = device.malloc(storageSize);
    auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

    TensorStorage storage(ptr, refPtr, storageSize, &device);

    if (type.id == DType::Float32) {
        auto generator = [&]() -> float {
            return rand() % 50;
        };

        std::generate((float*)cpuptr, (float*)cpuptr + shape.size(), generator);
    } else if (type.id == DType::Float64) {
        auto generator = [&]() -> double {
            return rand() % 50;
        };

        std::generate((double*)cpuptr, (double*)cpuptr + shape.size(), generator);
    } else {
        DEEP8_RUNTIME_ERROR("type is error");
    }

    /**copy to GPU*/
    device.copyFromCPUToGPU(cpuptr, ptr, storageSize);

    return Tensor(storage, 0, shape, type);
}
#endif

void zeroTensor(CPUDevice &device, Tensor &t) {
    device.zero(t.raw(), t.byteCount());
}

#ifdef HAVE_CUDA
void zeroTensor(GPUDevice& device, Tensor& t) {
    device.zero(t.raw(), t.byteCount());
}
#endif

Deep8::Variable createFakeVariable(CPUDevice &device, ElementType type) {
	Shape shape(1, {1});

	auto storageSize = type.byteWidth * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	Tensor value(storage, 0, shape, type);
	Tensor grad(storage, 0, shape, type);

	return Deep8::Variable(0, "", nullptr, value, grad);
}

Deep8::Variable createFakeVariable(CPUDevice &device, ElementType type, size_t batch, std::vector<size_t> list) {
	Shape shape(batch, list);

	auto storageSize = type.byteWidth * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	Tensor value(storage, 0, shape, type);
	Tensor grad(storage, 0, shape, type);

	return Deep8::Variable(0, "", nullptr, value, grad);
}

#ifdef HAVE_CUDA
Deep8::Variable createFakeVariable(GPUDevice& device, ElementType type) {
    Shape shape(1, { 1 });

    auto storageSize = type.byteWidth * shape.size();

    auto ptr = device.malloc(storageSize);
    auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

    TensorStorage storage(ptr, refPtr, storageSize, &device);

    Tensor value(storage, 0, shape, type);
    Tensor grad(storage, 0, shape, type);

    return Deep8::Variable(0, "", nullptr, value, grad);
}

Deep8::Variable createFakeVariable(GPUDevice& device, ElementType type, size_t batch, std::vector<size_t> list) {
    Shape shape(batch, list);

    auto storageSize = type.byteWidth * shape.size();

    auto ptr = device.malloc(storageSize);
    auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

    TensorStorage storage(ptr, refPtr, storageSize, &device);

    Tensor value(storage, 0, shape, type);
    Tensor grad(storage, 0, shape, type);

    return Deep8::Variable(0, "", nullptr, value, grad);
}
#endif


}

#endif //DEEP8_TESTUTILS_H
