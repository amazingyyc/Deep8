#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "Abs.h"

namespace Deep8 {

template <typename real>
__global__ void AbsForwardKernel(const real *x, real *y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		y[i] = cuAbs(x[i]);
	}
}

template <typename real>
__global__ void AbsBackwardKernel(const real *x, real *xGrad, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		if (x[i] > real(0)) {
			xGrad[i] += yGrad[i];
		} else if (x[i] < real(0)) {
			xGrad[i] -= yGrad[i];
		}
	}
}

template <typename T>
void Abs<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto x = inputs[0]->data();
	auto y = output->data();
	const int N = (int)output->shape.size();

	int minGrideSize;
	int blockSize;
	int grideSize;

	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AbsForwardKernel<T>, 0, N));

	grideSize = (N + blockSize - 1) / blockSize;

	AbsForwardKernel<T> << <grideSize, blockSize >> > (x, y, N);
}

#ifdef HAVE_HALF
template <>
void Abs<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
	auto x = inputs[0]->data();
	auto y = output->data();
	const int N = (int)output->shape.size();

	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	AbsForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
}
#endif

template <typename T>
void Abs<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
						const Tensor<T> *output,
						const Tensor<T> *outputGradient,
						size_t index,
						Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backward is error!");

	auto x = inputs[0]->data();
	auto dx = iGradient->data();
	auto dy = outputGradient->data();

	const int N = (int)iGradient->shape.size();

	int minGrideSize;
	int blockSize;
	int grideSize;

	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AbsBackwardKernel<T>, 0, N));

	grideSize = (N + blockSize - 1) / blockSize;

	AbsBackwardKernel<T> << <grideSize, blockSize >> > (x, dx, dy, N);
}

#ifdef HAVE_HALF
template <>
void Abs<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
							const Tensor<half> *output,
							const Tensor<half> *outputGradient,
							size_t index,
							Tensor<half> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backward is error!");

	auto x = inputs[0]->data();
	auto dx = iGradient->data();
	auto dy = outputGradient->data();

	const int N = (int)iGradient->shape.size();

	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	AbsBackwardKernel<half> << <grideSize, blockSize >> > (x, dx, dy, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(Abs);

}