#include "GPUDevice.h"
#include "Executor.h"

namespace Deep8 {

template <typename T>
void Executor<T>::initDeviceGPU() {
#ifdef HAVE_CUDA
	device = new GPUDevice();
#else
	DEEP8_RUNTIME_ERROR("not find a GPU");
#endif
}

template void Executor<float>::initDeviceGPU();
template void Executor<double>::initDeviceGPU();
#ifdef HAVE_HALF
template void Executor<half>::initDeviceGPU();
#endif

}