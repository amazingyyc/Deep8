#include "CudaHeads.h"
#include "CudaException.h"
#include "Tensor.h"

namespace Deep8 {

template <typename T>
T Tensor<T>::scalarGPU() {
#ifdef HAVE_CUDA
	T scalar;

	static_cast<GPUDevice*>(device)->copyFromGPUToCPU(raw(), &scalar, sizeof(T));

	return scalar;
#else
	DEEP8_RUNTIME_ERROR("can not call GPU function withou a GPU");
#endif // HAVE_CUDA
}

template void Tensor<float>::scalarGPU();
template void Tensor<double>::scalarGPU();

}