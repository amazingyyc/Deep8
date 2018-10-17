#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "MatrixMultiply.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <>
void MatrixMultiply<float>::forwardGPUImpl(Device *d, const float *A, const Shape &aShape, const float *B, const Shape &bShape, float *C, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	float alpha = 1;
	float beta  = 0;

	if (1 == bShape.batch()) {
		int m = aShape.batch() *aShape.row();
		int n = bShape.col();
		int k = aShape.col();

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int row = aShape.row();
		int col = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, row, b, col, &alpha, A, col, B, col, &beta, C, row));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto ABatch = A + (b % aShape.batch()) * aShape.batchSize();
			auto BBatch = B + (b % bShape.batch()) * bShape.batchSize();
			auto CBatch = C + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, BBatch, n, ABatch, k, &beta, CBatch, n));
		}
	}
}

template <>
void MatrixMultiply<double>::forwardGPUImpl(Device* d, const double *A, const Shape &aShape, const double *B, const Shape &bShape, double *C, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	double alpha = 1;
	double beta = 0;

	if (1 == bShape.batch()) {
		int m = aShape.batch() *aShape.row();
		int n = bShape.col();
		int k = aShape.col();

		CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int row = aShape.row();
		int col = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, row, b, col, &alpha, A, col, B, col, &beta, C, row));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto aPtr = A + (b % aShape.batch()) * aShape.batchSize();
			auto bPtr = B + (b % bShape.batch()) * bShape.batchSize();
			auto cPtr = C + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, bPtr, n, aPtr, k, &beta, cPtr, n));
		}
	}
}


#ifdef HAVE_HALF
template <>
void MatrixMultiply<half>::forwardGPUImpl(Device* d, const half *A, const Shape &aShape, const half *B, const Shape &bShape, half *C, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	half alpha = 1.0;
	half beta = 0.0;

	if (1 == bShape.batch()) {
		int m = aShape.batch() *aShape.row();
		int n = bShape.col();
		int k = aShape.col();

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int row = aShape.row();
		int col = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, row, b, col, &alpha, A, col, B, col, &beta, C, row));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto ABatch = A + (b % aShape.batch()) * aShape.batchSize();
			auto BBatch = B + (b % bShape.batch()) * bShape.batchSize();
			auto CBatch = C + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, BBatch, n, ABatch, k, &beta, CBatch, n));
		}
	}
}
#endif

template <typename T>
void MatrixMultiply<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	forwardGPUImpl(output->device(), inputs[0]->data(), inputs[0]->shape, inputs[1]->data(), inputs[1]->shape, output->data(), output->shape);
}

template <>
void MatrixMultiply<float>::backwardGPUImpl0(Device* d, float *aGrad, const Shape &aShape, const float *B, const Shape &bShape, const float *cGrad, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	float alpha = 1;
	float beta = 1;

	if (1 == bShape.batch()) {
		int b = aShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, b * m, n, &alpha, B, n, cGrad, n, &beta, aGrad, k));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int m = aShape.row();
		int k = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, m, b, &alpha, B, k, cGrad, m, &beta, aGrad, k));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto aGradPtr = aGrad + (b % aShape.batch()) * aShape.batchSize();
			auto bPtr = B + (b % bShape.batch()) * bShape.batchSize();
			auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, bPtr, n, cGradPtr, n, &beta, aGradPtr, k));
		}
	}
}

template <>
void MatrixMultiply<double>::backwardGPUImpl0(Device* d, double *aGrad, const Shape &aShape, const double *B, const Shape &bShape, const double *cGrad, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	double alpha = 1;
	double beta = 1;

	if (1 == bShape.batch()) {
		int b = aShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, b * m, n, &alpha, B, n, cGrad, n, &beta, aGrad, k));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int m = aShape.row();
		int k = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, m, b, &alpha, B, k, cGrad, m, &beta, aGrad, k));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto aGradPtr = aGrad + (b % aShape.batch()) * aShape.batchSize();
			auto bPtr = B + (b % bShape.batch()) * bShape.batchSize();
			auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, bPtr, n, cGradPtr, n, &beta, aGradPtr, k));
		}
	}
}

#ifdef HAVE_HALF
template <>
void MatrixMultiply<half>::backwardGPUImpl0(Device* d, half *aGrad, const Shape &aShape, const half *B, const Shape &bShape, const half *cGrad, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	half alpha = 1.0;
	half beta = 1.0;

	if (1 == bShape.batch()) {
		int b = aShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, b * m, n, &alpha, B, n, cGrad, n, &beta, aGrad, k));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int m = aShape.row();
		int k = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, m, b, &alpha, B, k, cGrad, m, &beta, aGrad, k));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto aGradPtr = aGrad + (b % aShape.batch()) * aShape.batchSize();
			auto bPtr = B + (b % bShape.batch()) * bShape.batchSize();
			auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, bPtr, n, cGradPtr, n, &beta, aGradPtr, k));
		}
	}
}
#endif

template <>
void MatrixMultiply<float>::backwardGPUImpl1(Device* d, const float *A, const Shape &aShape, float *bGrad, const Shape &bShape, const float *cGrad, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	float alpha = 1;
	float beta = 1;

	if (1 == bShape.batch()) {
		int b = aShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m * b, &alpha, cGrad, n, A, k, &beta, bGrad, n));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int m = aShape.row();
		int k = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, b, m, &alpha, A, k, cGrad, m, &beta, bGrad, k));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto aPtr = A + (b % aShape.batch()) * aShape.batchSize();
			auto bGradPtr = bGrad + (b % bShape.batch()) * bShape.batchSize();
			auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, cGradPtr, n, aPtr, k, &beta, bGradPtr, n));
		}
	}
}

template <>
void MatrixMultiply<double>::backwardGPUImpl1(Device* d, const double *A, const Shape &aShape, double *bGrad, const Shape &bShape, const double *cGrad, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	double alpha = 1;
	double beta = 1;

	if (1 == bShape.batch()) {
		int b = aShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m * b, &alpha, cGrad, n, A, k, &beta, bGrad, n));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int m = aShape.row();
		int k = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, b, m, &alpha, A, k, cGrad, m, &beta, bGrad, k));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto aPtr = A + (b % aShape.batch()) * aShape.batchSize();
			auto bGradPtr = bGrad + (b % bShape.batch()) * bShape.batchSize();
			auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, cGradPtr, n, aPtr, k, &beta, bGradPtr, n));
		}
	}
}

#ifdef HAVE_HALF
template <>
void MatrixMultiply<half>::backwardGPUImpl1(Device* d, const half *A, const Shape &aShape, half *bGrad, const Shape &bShape, const half *cGrad, const Shape &cShape) {
	auto device = (GPUDevice*)d;

	half alpha = 1.0;
	half beta = 1.0;

	if (1 == bShape.batch()) {
		int b = aShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m * b, &alpha, cGrad, n, A, k, &beta, bGrad, n));
	} else if (1 == aShape.batch() && 1 == bShape.col()) {
		int m = aShape.row();
		int k = aShape.col();
		int b = bShape.batch();

		CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, b, m, &alpha, A, k, cGrad, m, &beta, bGrad, k));
	} else {
		int batch = cShape.batch();
		int m = aShape.row();
		int k = aShape.col();
		int n = bShape.col();

		for (int b = 0; b < batch; ++b) {
			auto aPtr = A + (b % aShape.batch()) * aShape.batchSize();
			auto bGradPtr = bGrad + (b % bShape.batch()) * bShape.batchSize();
			auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

			CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, cGradPtr, n, aPtr, k, &beta, bGradPtr, n));
		}
	}
}
#endif

template <typename T>
void MatrixMultiply<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	auto device = static_cast<GPUDevice*>(iGradient->device());

	if (0 == index) {
		backwardGPUImpl0(device, iGradient->data(), iGradient->shape, inputs[1]->data(), inputs[1]->shape, outputGradient->data(), outputGradient->shape);
	} else if (1 == index) {
		backwardGPUImpl1(device, inputs[0]->data(), inputs[0]->shape, iGradient->data(), iGradient->shape, outputGradient->data(), outputGradient->shape);
	} else {
		DEEP8_RUNTIME_ERROR("the index is error");
	}
}

DEEP8_DECLARATION_GPU_FUNC(MatrixMultiply)

template void MatrixMultiply<float>::forwardGPUImpl(Device *device, const float *A, const Shape &aShape, const float *B, const Shape &bShape, float *C, const Shape &cShape);
template void MatrixMultiply<double>::forwardGPUImpl(Device *device, const double *A, const Shape &aShape, const double *B, const Shape &bShape, double *C, const Shape &cShape);

template void MatrixMultiply<float>::backwardGPUImpl0(Device* device, float *aGrad, const Shape &aShape, const float *B, const Shape &bShape, const float *cGrad, const Shape &cShape);
template void MatrixMultiply<double>::backwardGPUImpl0(Device* device, double *aGrad, const Shape &aShape, const double *B, const Shape &bShape, const double *cGrad, const Shape &cShape);

template void MatrixMultiply<float>::backwardGPUImpl1(Device* device, const float *A, const Shape &aShape, float *bGrad, const Shape &bShape, const float *cGrad, const Shape &cShape);
template void MatrixMultiply<double>::backwardGPUImpl1(Device* device, const double *A, const Shape &aShape, double *bGrad, const Shape &bShape, const double *cGrad, const Shape &cShape);

#ifdef HAVE_HALF
template void MatrixMultiply<half>::forwardGPUImpl(Device *device, const half *A, const Shape &aShape, const half *B, const Shape &bShape, half *C, const Shape &cShape);
template void MatrixMultiply<half>::backwardGPUImpl0(Device* device, half *aGrad, const Shape &aShape, const half *B, const Shape &bShape, const half *cGrad, const Shape &cShape);
template void MatrixMultiply<half>::backwardGPUImpl1(Device* device, const half *A, const Shape &aShape, half *bGrad, const Shape &bShape, const half *cGrad, const Shape &cShape);

#endif

#endif

}