#ifndef DEEP8_TESTUTILS_H
#define DEEP8_TESTUTILS_H

#include <stdarg.h>

namespace Deep8 {

#if HAVE_CUDA

template <typename T>
Tensor<T> createTensorGPU(GPUDevice *device, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
	Shape shape({ dim0, dim1, dim2, dim3 });
	auto gpuPtr = (T*)device->malloc(sizeof(T) * shape.size());

	return Tensor<T>(gpuPtr, shape, device);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice *device, size_t dim0, size_t dim1, size_t dim2) {
	Shape shape({ dim0, dim1, dim2 });
	auto gpuPtr = (T*)device->malloc(sizeof(T) * shape.size());

	return Tensor<T>(gpuPtr, shape, device);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice *device, size_t dim0, size_t dim1) {
	Shape shape({ dim0, dim1 });
	auto gpuPtr = (T*)device->malloc(sizeof(T) * shape.size());

	return Tensor<T>(gpuPtr, shape, device);
}


template <typename T>
Tensor<T> createTensorGPU(GPUDevice *device, T *cpuPtr, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
	Shape shape({ dim0, dim1, dim2, dim3 });
	auto gpuPtr = (T*) device->malloc(sizeof(T) * shape.size());

	/**generator random value*/
	auto generator = [&]() -> T {
		return rand() % 50 - 25;
	};

	std::generate(cpuPtr, cpuPtr + shape.size(), generator);

	/**copy to GPU*/
	device->copyFromCPUToGPU(cpuPtr, gpuPtr, sizeof(T) * shape.size());

	return Tensor<T>(gpuPtr, shape, device);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice *device, T *cpuPtr, size_t dim0, size_t dim1, size_t dim2) {
	Shape shape({ dim0, dim1, dim2 });
	auto gpuPtr = (T*) device->malloc(sizeof(T) * shape.size());

	/**generator random value*/
	auto generator = [&]() -> T {
		return rand() % 50 - 25;
	};

	std::generate(cpuPtr, cpuPtr + shape.size(), generator);

	/**copy to GPU*/
	device->copyFromCPUToGPU(cpuPtr, gpuPtr, sizeof(T) * shape.size());

	return Tensor<T>(gpuPtr, shape, device);
}

template <typename T>
Tensor<T> createTensorGPU(GPUDevice *device, T *cpuPtr, size_t dim0, size_t dim1) {
	Shape shape({ dim0, dim1 });
	auto gpuPtr = (T*) device->malloc(sizeof(T) * shape.size());

	/**generator random value*/
	auto generator = [&]() -> T {
		return rand() % 50 - 25;
	};

	std::generate(cpuPtr, cpuPtr + shape.size(), generator);

	/**copy to GPU*/
	device->copyFromCPUToGPU(cpuPtr, gpuPtr, sizeof(T) * shape.size());

	return Tensor<T>(gpuPtr, shape, device);
}


#endif


template<typename DeviceType, typename T>
Tensor<T> createTensor(DeviceType &device, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
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
}

template<typename DeviceType, typename T>
Tensor<T> createTensor(DeviceType &device, size_t dim0, size_t dim1, size_t dim2) {
	Shape shape({dim0, dim1, dim2});

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	auto generator = [&]() -> T {
		return rand() % 50;
	};

	std::generate((T*)ptr, (T*)ptr + shape.size(), generator);

	return Tensor<T>(storage, 0, shape);
}

template<typename DeviceType, typename T>
Tensor<T> createTensor(DeviceType &device, size_t dim0, size_t dim1) {
	Shape shape({ dim0, dim1 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	auto generator = [&]() -> T {
		return rand() % 50;
	};

	std::generate((T*)ptr, (T*)ptr + shape.size(), generator);

	return Tensor<T>(storage, 0, shape);
}

template <typename DeviceType, typename T>
void freeTensor(DeviceType &device, Tensor<T> &t) {
}

template <typename DeviceType, typename T>
void zeroTensor(DeviceType &device, Tensor<T> &t) {
    device.zero(t.raw(), sizeof(T) * t.size());
}

/**
 * create a fake Variable for test
 */
template <typename DeviceType, typename T>
Deep8::Variable<T> createFakeVariable(DeviceType &device) {
}

/**
 * create a fake Variable for test
 */
template <typename DeviceType = CPUDevice, typename T>
Deep8::Variable<T> createFakeVariable(CPUDevice &device) {
	Shape shape({ 1, 1 });

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	Tensor<T> value(storage, 0, shape);
	Tensor<T> grad(storage, 0, shape);

	return Deep8::Parameter<T>(value, grad);
}


template <typename DeviceType, typename T>
Deep8::Variable<T> createFakeVariable(DeviceType &device, std::initializer_list<size_t> list) {
    
}


template <typename DeviceType = CPUDevice, typename T>
Deep8::Variable<T> createFakeVariable(CPUDevice &device, std::initializer_list<size_t> list) {
	Shape shape(list);

	auto storageSize = sizeof(T) * shape.size();

	auto ptr = device.malloc(storageSize);
	auto refPtr = (size_t*)device.malloc(sizeof(size_t));

	TensorStorage storage(ptr, refPtr, storageSize, &device);

	Tensor<T> value(storage, 0, shape);
	Tensor<T> grad(storage, 0, shape);

	return Deep8::Parameter<T>(value, grad);
}


template <typename T>
void freeFakeVariable(Deep8::Variable<T> &var) {

}

}

#endif //DEEP8_TESTUTILS_H
