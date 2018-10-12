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
	explicit CPUDevice();

	~CPUDevice();

	void* malloc(size_t size) override;

	void free(void *ptr) override;

	void zero(void *ptr, size_t size) override;

	void copy(const void *from, void *to, size_t size) override;

	void copyFromCPUToGPU(const void *from, void *to, size_t size) override;

	void copyFromGPUToCPU(const void *from, void *to, size_t size) override;

	void copyFromGPUToGPU(const void *from, void *to, size_t size) override;

	void *mallocCPU(size_t size) override;

	void freeCPU(void *ptr) override;

	void* gpuOneHalf() override;
	
	void* gpuOneFloat() override;

	void* gpuOneDouble() override;


	int getDeviceThreadNum();
};

}

#endif //MAIN_DEVICE_H
