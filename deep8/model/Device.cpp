#include "Device.h"

namespace Deep8 {

CPUDevice::CPUDevice() : Device(DeviceType::CPU), randGenerator(randomDevice()) {
	memoryPool = new CPUMemoryPool();

	auto threadNum = getDeviceThreadNum();
	eigenThreadPool = new Eigen::NonBlockingThreadPool(threadNum);
	eigenDevice = new Eigen::ThreadPoolDevice(eigenThreadPool, threadNum);
}

CPUDevice::~CPUDevice() {
	delete memoryPool;
	delete eigenDevice;
	delete eigenThreadPool;
}

void* CPUDevice::malloc(size_t size) {
	return memoryPool->malloc(size);
}

void CPUDevice::free(void *ptr) {
	memoryPool->free(ptr);
}

void CPUDevice::zero(void *ptr, size_t size) {
	memoryPool->zero(ptr, size);
}

void CPUDevice::copy(const void *from, void *to, size_t size) {
	memoryPool->copy(from, to, size);
}

void CPUDevice::copyFromCPUToGPU(const void *from, void *to, size_t size) {
	memoryPool->copyFromCPUToGPU(from, to, size);
}

void CPUDevice::copyFromGPUToCPU(const void *from, void *to, size_t size) {
	memoryPool->copyFromGPUToCPU(from, to, size);
}

void CPUDevice::copyFromGPUToGPU(const void *from, void *to, size_t size) {
	memoryPool->copyFromGPUToGPU(from, to, size);
}

void* CPUDevice::mallocCPU(size_t size) {
	DEEP8_RUNTIME_ERROR("can not call mallocCPU function from CPUDevice");
}

void CPUDevice::freeCPU(void *ptr) {
	DEEP8_RUNTIME_ERROR("can not call freeCPU function from CPUDevice");
}

void* CPUDevice::gpuOneHalf() {
	DEEP8_RUNTIME_ERROR("can not call gpuOneHalf function from CPUDevice");
}

void* CPUDevice::gpuOneFloat() {
	DEEP8_RUNTIME_ERROR("can not call gpuOneFloat function from CPUDevice");
}

void* CPUDevice::gpuOneDouble() {
	DEEP8_RUNTIME_ERROR("can not call gpuOneDouble function from CPUDevice");
}

int CPUDevice::getDeviceThreadNum() {
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

}