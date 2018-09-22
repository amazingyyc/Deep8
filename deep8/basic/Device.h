#ifndef DEEP8_DEVICE_H
#define DEEP8_DEVICE_H

namespace Deep8 {

/**
 * @brief the type of the device
 * for now only the CPU is supported
 */
enum class DeviceType {
    CPU,
    GPU,
};

class Device {
public:
    DeviceType type;

    explicit Device(DeviceType type): type(type) {
    }

    virtual ~Device() = default;

    virtual void* malloc(size_t size) = 0;
    virtual void free(void *ptr) = 0;
    virtual void zero(void *ptr, size_t size) = 0;
    virtual void copy(const void *from, void *to, size_t size) = 0;
};

class CPUDevice: public Device {
public:
    CPUMemoryPool *memoryPool;

    std::random_device randomDevice;
    std::mt19937 randGenerator;

    /**the Eigen device for using Eigen Tensor*/
    Eigen::NonBlockingThreadPool *eigenThreadPool;
    Eigen::ThreadPoolDevice *eigenDevice;

public:
    explicit CPUDevice(): Device(DeviceType::CPU), randGenerator(randomDevice()) {
        memoryPool = new CPUMemoryPool();

        auto threadNum  = getDeviceThreadNum();
        eigenThreadPool = new Eigen::NonBlockingThreadPool(threadNum);
        eigenDevice     = new Eigen::ThreadPoolDevice(eigenThreadPool, threadNum);
    }

    ~CPUDevice() {
        delete memoryPool;
        delete eigenDevice;
        delete eigenThreadPool;
    }

    void* malloc(size_t size) override {
        return memoryPool->malloc(size);
    }

    void free(void *ptr) override {
        memoryPool->free(ptr);
    }

    void zero(void *ptr, size_t size) override {
		memoryPool->zero(ptr, size);
    }

    void copy(const void *from, void *to, size_t size) override {
		memoryPool->copy(from, to, size);
    }
};

#ifdef HAVE_CUDA

class GPUDevice : public Device {
public:
	/**the GPU memory allocator*/
	GPUMemoryAllocator *gpuMemoryAllocator;


	/**the GPU device id*/
	int deviceId;

	/**cuBlas handle*/
	cublasHandle_t cublasHandle;

    /**cuRand generator*/
    curandGenerator_t curandGenerator;

#ifdef HAVE_CUDNN
	/**cudnn handle*/
	cudnnHandle_t cudnnHandle;
#endif

	/**the GPU memroy contains 1, 0, -1*/
	float *floatOne;
	float *floatZero;
	float *floatMinusOne;

	double *doubleOne;
	double *doubleZero;
	double *doubleMinusOne;

#ifdef HAVE_HALF
	half *halfOne;
	half *halfZero;
	half *halfMinusOne;
#endif // HAVE_HALF


	explicit GPUDevice() : GPUDevice(0) {
	}

	explicit GPUDevice(int deviceId) : Device(DeviceType::GPU), deviceId(deviceId) {
		CUDA_CHECK(cudaSetDevice(deviceId));

		gpuMemoryAllocator = new GPUMemoryAllocator(deviceId);
		
		CUBLAS_CHECK(cublasCreate(&cublasHandle));

		CURAND_CHECK(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGenerator, (unsigned long long)time(nullptr)));

#ifdef HAVE_CUDNN
		CUDNN_CHECK(cudnnCreate(&cudnnHandle));
#endif

#ifdef HAVE_HALF
		void *ptr = gpuMemoryAllocator->malloc(sizeof(float) * 3 + sizeof(double) * 3 + sizeof(half) * 3);

		floatOne = (float*)ptr;
		floatZero = floatOne + 1;
		floatMinusOne = floatZero + 1;

		doubleOne = (double*)(floatMinusOne + 1);
		doubleZero = doubleOne + 1;
		doubleMinusOne = doubleZero + 1;

		halfOne = (half*)(doubleMinusOne + 1);
		halfZero = halfOne + 1;
		halfMinusOne = halfZero + 1;

		float  numberF[3] = { 1, 0, -1 };
		double numberD[3] = { 1, 0, -1 };
		half   numberH[3] = { 1.0, 0.0, -1.0 };

		gpuMemoryAllocator->copyFromCPUToGPU(&numberF[0], floatOne,  sizeof(float)  * 3);
		gpuMemoryAllocator->copyFromCPUToGPU(&numberD[0], doubleOne, sizeof(double) * 3);
		gpuMemoryAllocator->copyFromCPUToGPU(&numberH[0], halfOne,   sizeof(half)   * 3);

#else
		void *ptr = gpuMemoryAllocator->malloc(sizeof(float) * 3 + sizeof(double) * 3);

		floatOne = (float*)ptr;
		floatZero = floatOne + 1;
		floatMinusOne = floatZero + 1;

		doubleOne = (double*)(floatMinusOne + 1);
		doubleZero = doubleOne + 1;
		doubleMinusOne = doubleZero + 1;

		float numberF[3] = { 1, 0, -1 };
		double numberD[3] = { 1, 0, -1 };

		gpuMemoryAllocator->copyFromCPUToGPU(&numberF[0], floatOne, sizeof(float) * 3);
		gpuMemoryAllocator->copyFromCPUToGPU(&numberD[0], doubleOne, sizeof(double) * 3);
#endif // HAVE_HALF
	}

	~GPUDevice() {
		gpuMemoryAllocator->free(floatOne);

		delete gpuMemoryAllocator;

		CUBLAS_CHECK(cublasDestroy(cublasHandle));
		CURAND_CHECK(curandDestroyGenerator(curandGenerator));

#ifdef HAVE_CUDNN
		CUDNN_CHECK(cudnnDestroy(cudnnHandle));
#endif
	}

	void* malloc(size_t size) override {
		return gpuMemoryAllocator->malloc(size);
	}

	void free(void *ptr) override {
		gpuMemoryAllocator->free(ptr);
	}

	void zero(void *ptr, size_t size) override {
		gpuMemoryAllocator->zero(ptr, size);
	}

	void copy(const void *from, void *to, size_t size) override {
		gpuMemoryAllocator->copy(from, to, size);
	}

	void copyFromCPUToGPU(const void *from, void *to, size_t size) {
		gpuMemoryAllocator->copyFromCPUToGPU(from, to, size);
	}

	void copyFromGPUToCPU(const void *from, void *to, size_t size) {
		gpuMemoryAllocator->copyFromGPUToCPU(from, to, size);
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) {
		gpuMemoryAllocator->copyFromGPUToGPU(from, to, size);
	}

	template<typename T>
	void *gpuOne() {
		DEEP8_RUNTIME_ERROR("get GPU number value error");
	}

	template <>
	void *gpuOne<float>() {
		return floatOne;
	}

	template <>
	void *gpuOne<double>() {
		return doubleOne;
	}

#ifdef HAVE_HALF
	template <>
	void *gpuOne<half>() {
		return halfOne;
	}
#endif // HAVE_HALF


	template<typename T>
	void *gpuZero() {
		DEEP8_RUNTIME_ERROR("get GPU number value error");
	}

	template <>
	void *gpuZero<float>() {
		return floatZero;
	}

	template <>
	void *gpuZero<double>() {
		return doubleZero;
	}

#ifdef HAVE_HALF
	template <>
	void *gpuZero<half>() {
		return halfZero;
	}
#endif // HAVE_HALF

	template<typename T>
	void *gpuMinusOne() {
		DEEP8_RUNTIME_ERROR("get GPU number value error");
	}

	template <>
	void *gpuMinusOne<float>() {
		return floatMinusOne;
	}

	template <>
	void *gpuMinusOne<double>() {
		return doubleMinusOne;
	}

#ifdef HAVE_HALF
	template <>
	void *gpuMinusOne<half>() {
		return halfMinusOne;
	}
#endif // HAVE_HALF

};

#endif

}

#endif //MAIN_DEVICE_H
