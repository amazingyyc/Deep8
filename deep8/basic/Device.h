#ifndef DEEP8_DEVICE_H
#define DEEP8_DEVICE_H

#include "Basic.h"
#include "MemoryPool.h"

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
	virtual void copyFromCPUToGPU(const void *from, void *to, size_t size) = 0;
	virtual void copyFromGPUToCPU(const void *from, void *to, size_t size) = 0;
	virtual void copyFromGPUToGPU(const void *from, void *to, size_t size) = 0;

	virtual void* mallocCPU(size_t size) = 0;
	virtual void freeCPU(void *ptr) = 0;

	virtual void* gpuOneHalf()   = 0;
	virtual void* gpuOneFloat()  = 0;
	virtual void* gpuOneDouble() = 0;
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

	void copyFromCPUToGPU(const void *from, void *to, size_t size) override {
		memoryPool->copyFromCPUToGPU(from, to, size);
	}

	void copyFromGPUToCPU(const void *from, void *to, size_t size) override {
		memoryPool->copyFromGPUToCPU(from, to, size);
	}

	void copyFromGPUToGPU(const void *from, void *to, size_t size) override {
		memoryPool->copyFromGPUToGPU(from, to, size);
	}

	void *mallocCPU(size_t size) override {
		DEEP8_RUNTIME_ERROR("can not call mallocCPU function from CPUDevice");
	}

	void freeCPU(void *ptr) override {
		DEEP8_RUNTIME_ERROR("can not call freeCPU function from CPUDevice");
	}

	void* gpuOneHalf() override {
		DEEP8_RUNTIME_ERROR("can not call gpuOneHalf function from CPUDevice");
	}
	
	void* gpuOneFloat() override {
		DEEP8_RUNTIME_ERROR("can not call gpuOneFloat function from CPUDevice");
	}

	void* gpuOneDouble() override {
		DEEP8_RUNTIME_ERROR("can not call gpuOneDouble function from CPUDevice");
	}

protected:

	int getDeviceThreadNum() {
		int threadNum = 0;

#ifdef __GNUC__
		threadNum = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
#elif defined(_MSC_VER)
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		threadNum = static_cast<int>(info.dwNumberOfProcessors);
#else
		DEEP8_RUNTIME_ERROR("the compile not supported!");
#endif

		if (0 >= threadNum) {
			threadNum = 4;
		}

		return threadNum;
	}
};

}

#endif //MAIN_DEVICE_H
