#ifndef DEEP8_MATRIXMULTIPLY_H
#define DEEP8_MATRIXMULTIPLY_H

#include "Tensor.h"
#include "Shape.h"
#include "Function.h"

namespace Deep8 {

template <typename T>
class MatrixMultiply: public Function<T> {
public:
    explicit MatrixMultiply(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
    }

    /**
     * @brief for the MatrixMultiply the input size must be 2, and must be Matrix
     * @param inputs the inputs Node must be
     * @return the output Shape
     */
     void check() override {
         Function<T>::check();

         DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs dim must be 2");

         auto xValue = static_cast<Variable<T>*>(this->inputs[0])->value;
         auto yValue = static_cast<Variable<T>*>(this->inputs[1])->value;

         auto xShape = xValue.shape;
         auto yShape = yValue.shape;

         DEEP8_ARGUMENT_CHECK(xShape.batch() == yShape.batch() || 1 == xShape.batch() || 1 == yShape.batch(), "the batch of input is error");
         DEEP8_ARGUMENT_CHECK((2 == xShape.nDims() || 3 == xShape.nDims()) && (2 == yShape.nDims() || 3 == yShape.nDims()), "the inputs dimensions is error");
         DEEP8_ARGUMENT_CHECK(xShape.col() == yShape.row(), "the col of input1 must same to the row of input2");

         this->outputShape = Shape({std::max<size_t>(xShape.batch(), yShape.batch()), xShape.row(), xShape.col()});
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
         auto xTensor = inputs[0];
         auto yTensor = inputs[1];

         if (1 == yTensor->batch()) {
             eRowBatchMat(output).noalias() = eRowBatchMat(xTensor) * eMat(yTensor);
         } else if (1 == xTensor->batch() && 1 == yTensor->col()) {
             eBatchSizeMat(output).noalias() = eBatchSizeMat(yTensor) * eMat(xTensor).transpose();
         } else {
             DEEP8_ARGUMENT_CHECK(1 == xTensor->batch() || xTensor->batch() == yTensor->batch(), "the inputs batch error");
             DEEP8_ARGUMENT_CHECK(std::max<size_t>(xTensor->batch(), yTensor->batch()) == output->batch(), "the output batch is error");

             for (size_t b = 0; b < output->batch(); ++b) {
                 eBatchMat(output, b).noalias() = eBatchMat(xTensor, b) * eBatchMat(yTensor, b);
             }
         }
    }

    

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        if (0 == index) {
            /**
             * for a MatrixMultiply C = A * B, index is 0 means calculate the grad for A
             * grad(A) = grad(C) * transpose(B)
             */
            if (1 == inputs[1]->batch()) {
                eRowBatchMat(iGradient).noalias() += eRowBatchMat(outputGradient) * eMat(inputs[1]).transpose();
            } else if (1 == inputs[0]->batch() && 1 == inputs[1]->col()) {
                eMat(iGradient).noalias() += eBatchSizeMat(outputGradient).transpose() * eBatchSizeMat(inputs[1]);
            } else {
                for (size_t b = 0; b < outputGradient->batch(); ++b) {
                    eBatchMat(iGradient, b).noalias() += eBatchMat(outputGradient, b) * eBatchMat(inputs[1], b).transpose();
                }
            }
        } else if (1 == index) {
            /**
             * for a MatrixMultiply C = A * B, index is 1 means calculate the grad for B
             * grad(B) = transpose(A) * grad(C)
             */
			 if (1 == iGradient->batch()) {
				 eMat(iGradient).noalias() += eRowBatchMat(inputs[0]).transpose() * eRowBatchMat(outputGradient);
			 } else if (1 == inputs[0]->batch() && 1 == inputs[1]->col()) {
			     eBatchSizeMat(iGradient).noalias() += eBatchSizeMat(outputGradient) * eMat(inputs[0]);
			 } else {
                for (size_t b = 0; b < outputGradient->batch(); ++b) {
                    eBatchMat(iGradient, b).noalias() += eBatchMat(inputs[0], b).transpose() * eBatchMat(outputGradient, b);
                }
            }
        }
    }

#ifdef HAVE_CUDA

    /**
     * the Matrix in Deep8 is row-major
     */
    void forwardGPUImpl(GPUDevice *device, const float *A, const Shape &aShape, const float *B, const Shape &bShape, float *C, const Shape &cShape) {
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
            int b   = bShape.batch();

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

                CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, BBatch , n, ABatch, k, &beta, CBatch, n));
            }
        }
    }

    void forwardGPUImpl(GPUDevice *device, const double *A, const Shape &aShape, const double *B, const Shape &bShape, double *C, const Shape &cShape) {
        double alpha = 1;
        double beta  = 0;

        if (1 == bShape.batch()) {
            int m = aShape.batch() *aShape.row();
            int n = bShape.col();
            int k = aShape.col();

            CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
        } else if (1 == aShape.batch() && 1 == bShape.col()) {
            int row = aShape.row();
            int col = aShape.col();
            int b   = bShape.batch();

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

                CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, bPtr , n, aPtr, k, &beta, cPtr, n));
            }
        }
    }

#endif

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
    forwardGPUImpl(static_cast<GPUDevice*>(output->device), 
                    inputs[0]->data(), inputs[0]->shape, 
                    inputs[1]->data(), inputs[1]->shape, 
                    output->data(), output->shape);
#else
    DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }

#ifdef HAVE_CUDA

    void backwardGPUImpl0(GPUDevice* device, float *aGrad, const Shape &aShape, const float *B, const Shape &bShape, const float *cGrad, const Shape &cShape) {
        float alpha = 1;
        float beta  = 1;

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
                auto bPtr     = B     + (b % bShape.batch()) * bShape.batchSize();
                auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

                CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, bPtr, n, cGradPtr, n, &beta, aGradPtr, k));
            }
        }
    }

    void backwardGPUImpl0(GPUDevice* device, double *aGrad, const Shape &aShape, const double *B, const Shape &bShape, const double *cGrad, const Shape &cShape) {
        double alpha = 1;
        double beta  = 1;

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
                auto bPtr     = B     + (b % bShape.batch()) * bShape.batchSize();
                auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

                CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, bPtr, n, cGradPtr, n, &beta, aGradPtr, k));
            }
        }
    }

    void backwardGPUImpl1(GPUDevice* device, const float *A, const Shape &aShape, float *bGrad, const Shape &bShape, const float *cGrad, const Shape &cShape) {
        float alpha = 1;
        float beta  = 1;

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
                auto aPtr     = A     + (b % aShape.batch()) * aShape.batchSize();
                auto bGradPtr = bGrad + (b % bShape.batch()) * bShape.batchSize();
                auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

                CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, cGradPtr, n, aPtr, k, &beta, bGradPtr, n));
            }
        }
    }

    void backwardGPUImpl1(GPUDevice* device, const double *A, const Shape &aShape, double *bGrad, const Shape &bShape, const double *cGrad, const Shape &cShape) {
        double alpha = 1;
        double beta  = 1;

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
                auto aPtr     = A     + (b % aShape.batch()) * aShape.batchSize();
                auto bGradPtr = bGrad + (b % bShape.batch()) * bShape.batchSize();
                auto cGradPtr = cGrad + (b % cShape.batch()) * cShape.batchSize();

                CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, cGradPtr, n, aPtr, k, &beta, bGradPtr, n));
            }
        }
    }
#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
    auto device = static_cast<GPUDevice*>(iGradient->device);

    if (0 == index) {
        backwardGPUImpl0(device, iGradient->data(), iGradient->shape, inputs[1]->data(), inputs[1]->shape, outputGradient->data(), outputGradient->shape);
    } else if (1 == index) {
        backwardGPUImpl1(device, inputs[0]->data(), inputs[0]->shape, iGradient->data(), iGradient->shape, outputGradient->data(), outputGradient->shape);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
#else
    DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_MATRIXMULTIPLY_H
