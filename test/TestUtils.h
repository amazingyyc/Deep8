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
Tensor<T> createTensor(DeviceType *device, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
	Shape shape({ dim0, dim1, dim2, dim3 });

	auto ptr = (T*)device->malloc(sizeof(T) * shape.size());

	auto generator = [&]() -> T {
		return rand() % 50;
	};

	std::generate(ptr, ptr + shape.size(), generator);

	return Tensor<T>(ptr, shape, device);
}

template<typename DeviceType, typename T>
Tensor<T> createTensor(DeviceType *device, size_t dim0, size_t dim1, size_t dim2) {
	Shape shape({dim0, dim1, dim2});

	auto ptr = (T*)device->malloc(sizeof(T) * shape.size());

	auto generator = [&]() -> T {
		return rand() % 50;
	};

	std::generate(ptr, ptr + shape.size(), generator);

	return Tensor<T>(ptr, shape, device);
}

template<typename DeviceType, typename T>
Tensor<T> createTensor(DeviceType *device, size_t dim0, size_t dim1) {
	Shape shape({ dim0, dim1 });

	auto ptr = (T*)device->malloc(sizeof(T) * shape.size());

	auto generator = [&]() -> T {
		return rand() % 50;
	};

	std::generate(ptr, ptr + shape.size(), generator);

	return Tensor<T>(ptr, shape, device);
}

template <typename DeviceType, typename T>
void freeTensor(DeviceType *device, Tensor<T> &t) {
    device->free(t.pointer);
}

template <typename DeviceType, typename T>
void zeroTensor(DeviceType *device, Tensor<T> &t) {
    device->zero(t.pointer, sizeof(T) * t.size());
}

/**
 * create a fake Variable for test
 */
template <typename DeviceType, typename T>
Deep8::Variable<T> createFakeVariable(DeviceType *device) {
    Shape shape({1, 1});

    auto value = Tensor<T>(nullptr, shape, device);
    auto grad  = Tensor<T>(nullptr, shape, device);

    return Deep8::Parameter<T>(value, grad);
}

template <typename DeviceType, typename T>
Deep8::Variable<T> createFakeVariable(DeviceType *device, std::initializer_list<size_t> list) {
    Shape shape(list);

    auto value = Tensor<T>(nullptr, shape, device);
    auto grad  = Tensor<T>(nullptr, shape, device);

    return Deep8::Parameter<T>(value, grad);
}

template <typename T>
void freeFakeVariable(Deep8::Variable<T> &var) {

}

}

#endif //DEEP8_TESTUTILS_H
