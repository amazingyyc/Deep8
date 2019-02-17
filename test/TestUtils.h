#ifndef DEEP8_TESTUTILS_H
#define DEEP8_TESTUTILS_H

#include <stdarg.h>

#include "model/ElementType.h"
#include "model/Shape.h"
#include "model/Tensor.h"
#include "nodes/Variable.h"
#include "nodes/Parameter.h"

namespace Deep8 {

#if HAVE_CUDA

/**
Shape shape({ dim0, dim1, dim2, dim3 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	auto generator = [&]() -> T {
		return rand() % 50;
	};

	std::generate((T*)ptr, (T*)ptr + shape.size(), generator);

	return Tensor<T>(storage, 0, shape);
*/
template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4) {
	std::vector<size_t> list({ dim0, dim1, dim2, dim3 });
	Shape shape(dim0, { dim1, dim2, dim3, dim4 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
	std::vector<size_t> list({ dim0, dim1, dim2, dim3 });
	Shape shape(dim0, {dim1, dim2, dim3});

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, size_t dim0, size_t dim1, size_t dim2) {
	std::vector<size_t> list({ dim0, dim1, dim2});
	Shape shape(dim0, { dim1, dim2 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, size_t dim0, size_t dim1) {
	std::vector<size_t> list({ dim0, dim1});
	Shape shape(dim0, { dim1 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, T *cpuPtr, size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4) {
	std::vector<size_t> list({ dim0, dim1, dim2, dim3 });
	Shape shape(dim0, { dim1, dim2, dim3, dim4 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	/**generator random value*/
	auto generator = [&]() -> T {
		return rand() % 50 - 25;
	};

	std::generate(cpuPtr, cpuPtr + shape.size(), generator);

	/**copy to GPU*/
	device.copyFromCPUToGPU(cpuPtr, ptr, sizeof(T) * shape.size());

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, T *cpuPtr, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
	std::vector<size_t> list({ dim0, dim1, dim2, dim3});
	Shape shape(dim0, { dim1, dim2, dim3 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	/**generator random value*/
	auto generator = [&]() -> T {
		return rand() % 50 - 25;
	};

	std::generate(cpuPtr, cpuPtr + shape.size(), generator);

	/**copy to GPU*/
	device.copyFromCPUToGPU(cpuPtr, ptr, sizeof(T) * shape.size());

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, T *cpuPtr, size_t dim0, size_t dim1, size_t dim2) {
	std::vector<size_t> list({ dim0, dim1, dim2});
	Shape shape(dim0, {dim1, dim2});

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	/**generator random value*/
	auto generator = [&]() -> T {
		return rand() % 50 - 25;
	};

	std::generate(cpuPtr, cpuPtr + shape.size(), generator);

	/**copy to GPU*/
	device.copyFromCPUToGPU(cpuPtr, ptr, sizeof(T) * shape.size());

	return Tensor<T>(storage, 0, shape);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice &device, T *cpuPtr, size_t dim0, size_t dim1) {
	std::vector<size_t> list({ dim0, dim1 });
	Shape shape(dim0, { dim1 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	/**generator random value*/
	auto generator = [&]() -> T {
		return rand() % 50 - 25;
	};

	std::generate(cpuPtr, cpuPtr + shape.size(), generator);

	/**copy to GPU*/
	device.copyFromCPUToGPU(cpuPtr, ptr, sizeof(T) * shape.size());

	return Tensor<T>(storage, 0, shape);
}


#endif


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

// template<typename DeviceType, typename T>
// Tensor<T> createTensor(DeviceType &device, size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4) {
// 	Shape shape(dim0, { dim1 , dim2, dim3, dim4 });

// 	auto storageSize = sizeof(T) * shape.size();


// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	auto generator = [&]() -> T {
// 		return rand() % 50;
// 	};

// 	std::generate((T*)ptr, (T*)ptr + shape.size(), generator);

// 	return Tensor<T>(storage, 0, shape);
// }

// template<typename DeviceType, typename T>
// Tensor<T> createTensor(DeviceType &device, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
// 	// std::vector<size_t> list({ dim0, dim1, dim2, dim3 });
// 	Shape shape(dim0, { dim1 , dim2, dim3});

// 	auto storageSize = sizeof(T) * shape.size();

// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	auto generator = [&]() -> T {
// 		return rand() % 50;
// 	};

// 	std::generate((T*)ptr, (T*)ptr + shape.size(), generator);

// 	return Tensor<T>(storage, 0, shape);
// }

// template<typename DeviceType, typename T>
// Tensor<T> createTensor(DeviceType &device, size_t dim0, size_t dim1, size_t dim2) {
// 	// std::vector<size_t> list({ dim0, dim1, dim2});
// 	Shape shape(dim0, {dim1, dim2});

// 	auto storageSize = sizeof(T) * shape.size();

// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	auto generator = [&]() -> T {
// 		return rand() % 50;
// 	};

// 	std::generate((T*)ptr, (T*)ptr + shape.size(), generator);

// 	return Tensor<T>(storage, 0, shape);
// }

// template<typename DeviceType, typename T>
// Tensor<T> createTensor(DeviceType &device, size_t dim0, size_t dim1) {
// 	// std::vector<size_t> list({ dim0, dim1});
// 	Shape shape(dim0, {dim1});

// 	auto storageSize = sizeof(T) * shape.size();

// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	auto generator = [&]() -> T {
// 		return rand() % 50;
// 	};

// 	std::generate((T*)ptr, (T*)ptr + shape.size(), generator);

// 	return Tensor<T>(storage, 0, shape);
// }

// template <typename DeviceType, typename T>
// void freeTensor(DeviceType &device, Tensor<T> &t) {
// }

void zeroTensor(CPUDevice &device, Tensor &t) {
    device.zero(t.raw(), t.byteCount());
}

Deep8::Variable createFakeVariable(CPUDevice &device, ElementType type) {
	Shape shape(1, {1});

	auto storageSize = type.byteWidth * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	Tensor value(storage, 0, shape, type);
	Tensor grad(storage, 0, shape, type);

	return Deep8::Parameter(value, grad);
}

Deep8::Variable createFakeVariable(CPUDevice &device, ElementType type, size_t batch, std::vector<size_t> list) {
	Shape shape(batch, list);

	auto storageSize = type.byteWidth * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	Tensor value(storage, 0, shape, type);
	Tensor grad(storage, 0, shape, type);

	return Deep8::Parameter(value, grad);
}

// /**
//  * create a fake Variable for test
//  */
// template <typename DeviceType, typename T>
// Deep8::Variable<T> createFakeVariable(DeviceType &device) {
// }

// /**
//  * create a fake Variable for test
//  */
// template <typename DeviceType = CPUDevice, typename T>
// Deep8::Variable<T> createFakeVariable(CPUDevice &device) {
// 	std::vector<size_t> list({1, 1});
// 	Shape shape(1, {1});

// 	auto storageSize = sizeof(T) * shape.size();

// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	Tensor<T> value(storage, 0, shape);
// 	Tensor<T> grad(storage, 0, shape);

// 	return Deep8::Parameter<T>(value, grad);
// }

// #ifdef HAVE_CUDA
// template <typename DeviceType = GPUDevice, typename T>
// Deep8::Variable<T> createFakeVariable(GPUDevice &device) {
// 	std::vector<size_t> list({1, 1});
// 	Shape shape(1, {1});

// 	auto storageSize = sizeof(T) * shape.size();

// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	Tensor<T> value(storage, 0, shape);
// 	Tensor<T> grad(storage, 0, shape);

// 	return Deep8::Parameter<T>(value, grad);
// }
// #endif

// template <typename DeviceType, typename T>
// Deep8::Variable<T> createFakeVariable(DeviceType &device, std::vector<size_t> list) {
    
// }


// template <typename DeviceType = CPUDevice, typename T>
// Deep8::Variable<T> createFakeVariable(CPUDevice &device, std::vector<size_t> list) {
// 	auto batch = list[0];
// 	list.erase(list.begin());
// 	Shape shape(batch, list);

// 	auto storageSize = sizeof(T) * shape.size();

// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	Tensor<T> value(storage, 0, shape);
// 	Tensor<T> grad(storage, 0, shape);

// 	return Deep8::Parameter<T>(value, grad);
// }

// #ifdef HAVE_CUDA
// template <typename DeviceType = GPUDevice, typename T>
// Deep8::Variable<T> createFakeVariable(GPUDevice &device, std::initializer_list<size_t> list) {
// 	std::vector<size_t> ll(list);
// 	auto batch = ll[0];
// 	ll.erase(ll.begin());
// 	Shape shape(batch, ll);

// 	auto storageSize = sizeof(T) * shape.size();

// 	auto ptr = device.malloc(storageSize);
// 	auto refPtr = (size_t*)device.mallocCPU(sizeof(size_t));

// 	TensorStorage storage(ptr, refPtr, storageSize, &device);

// 	Tensor<T> value(storage, 0, shape);
// 	Tensor<T> grad(storage, 0, shape);

// 	return Deep8::Parameter<T>(value, grad);
// }

// #endif

// template <typename T>
// void freeFakeVariable(Deep8::Variable<T> &var) {

// }

}

#endif //DEEP8_TESTUTILS_H
