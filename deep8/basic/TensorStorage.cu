#include "TensorStorage.h"

namespace Deep8 {

void TensorStorage::freeGPU() {
#ifdef HAVE_CUDA
	auto gpuDevice = static_cast<GPUDevice*>(device);

	gpuDevice->free(ptr);
	gpuDevice->freeCPU(refPtr);

	ptr    = nullptr;
	refPtr = nullptr;
	size   = 0;

	device = nullptr;
#else
	DEEP8_RUNTIME_ERROR("can not call GPU function withou a GPU");
#endif
}


}