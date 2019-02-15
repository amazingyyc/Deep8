#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/MatrixMultiply.h"

namespace Deep8 {
namespace Math {

template <typename T>
void MatrixMultiplyGPUImpl(GPUDevice* device, 
                           const T* x, 
                           const Shape& xshape, 
                           const T* y, 
                           const Shape& yshape, 
                           T* z, 
                           const Shape& zshape) {
    DEEP8_RUNTIME_ERROR("the type in not support");
}

template <>
void MatrixMultiplyGPUImpl<float>( GPUDevice* device,
                                   const float* x,
                                   const Shape& xshape,
                                   const float* y,
                                   const Shape& yshape,
                                   float* z,
                                   const Shape& zshape) {
    float alpha = 1;
    float beta  = 0;

    if (1 == yshape.batch) {
        int m = xshape.batch * xshape.row();
        int n = yshape.col();
        int k = xshape.col();

        CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, y, n, x, k, &beta, z, n));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int row = xshape.row();
		int col = xshape.col();
		int b   = yshape.batch;

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, row, b, col, &alpha, x, col, y, col, &beta, z, row));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
		int k = xshape.col();
        int n = yshape.col(); 
        
        for (int b = 0; b < batch; ++b) {
			auto xptr = x + (b % xshape.batch) * xshape.batchSize();
			auto yptr = y + (b % yshape.batch) * yshape.batchSize();
			auto zptr = z + (b % zshape.batch) * zshape.batchSize();

			CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, yptr, n, xptr, k, &beta, zptr, n));
		}
    }
}

template <>
void MatrixMultiplyGPUImpl<double>(GPUDevice* device,
                                    const double* x,
                                    const Shape& xshape,
                                    const double* y,
                                    const Shape& yshape,
                                    double* z,
                                    const Shape& zshape) {
    double alpha = 1;
    double beta  = 0;

    if (1 == yshape.batch) {
        int m = xshape.batch * xshape.row();
        int n = yshape.col();
        int k = xshape.col();

        CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, y, n, x, k, &beta, z, n));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int row = xshape.row();
        int col = xshape.col();
        int b   = yshape.batch;

        CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, row, b, col, &alpha, x, col, y, col, &beta, z, row));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col(); 
        
        for (int b = 0; b < batch; ++b) {
            auto xptr = x + (b % xshape.batch) * xshape.batchSize();
            auto yptr = y + (b % yshape.batch) * yshape.batchSize();
            auto zptr = z + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, yptr, n, xptr, k, &beta, zptr, n));
        }
    }
}

#ifdef HAVE_HALF
template <>
void MatrixMultiplyGPUImpl<half>(GPUDevice* device,
                                    const half* x,
                                    const Shape& xshape,
                                    const half* y,
                                    const Shape& yshape,
                                    half* z,
                                    const Shape& zshape) {
    half alpha = 1.0;
    half beta  = 0.0;

    if (1 == yshape.batch) {
        int m = xshape.batch * xshape.row();
        int n = yshape.col();
        int k = xshape.col();

        CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, y, n, x, k, &beta, z, n));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int row = xshape.row();
        int col = xshape.col();
        int b   = yshape.batch;

        CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, row, b, col, &alpha, x, col, y, col, &beta, z, row));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col(); 
        
        for (int b = 0; b < batch; ++b) {
            auto xptr = x + (b % xshape.batch) * xshape.batchSize();
            auto yptr = y + (b % yshape.batch) * yshape.batchSize();
            auto zptr = z + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, yptr, n, xptr, k, &beta, zptr, n));
        }
    }
}
#endif

void MatrixMultiplyGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = (GPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        MatrixMultiplyGPUImpl<float>(device,
                                    x.data<float>(),
                                    x.shape,
                                    y.data<float>(),
                                    y.shape,
                                    z.data<float>(),
                                    z.shape);
        break;
    case DType::Float64:
        MatrixMultiplyGPUImpl<double>(device,
                                    x.data<double>(),
                                    x.shape,
                                    y.data<double>(),
                                    y.shape,
                                    z.data<double>(),
                                    z.shape);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        MatrixMultiplyGPUImpl<half>(device,
                                    x.data<half>(),
                                    x.shape,
                                    y.data<half>(),
                                    y.shape,
                                    z.data<half>(),
                                    z.shape);
        break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

/**gradient for X*/
template <typename T>
void MatrixMultiplyGradXGPUImpl(GPUDevice *device, 
                                const T *x, 
                                T *dx, 
                                const Shape &xshape, 
                                const T *y, 
                                const Shape &yshape, 
                                const T *z, 
                                const T *dz, 
                                const Shape &zshape) {
    DEEP8_RUNTIME_ERROR("the type in not support");
}

template <>
void MatrixMultiplyGradXGPUImpl<float>( GPUDevice *device, 
                                        const float *x, 
                                        float *dx, 
                                        const Shape &xshape, 
                                        const float *y, 
                                        const Shape &yshape, 
                                        const float *z, 
                                        const float *dz, 
                                        const Shape &zshape) {
    float alpha = 1;
    float beta  = 1;

    if (1 == yhape.batch) {
        int b = xshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, b * m, n, &alpha, y, n, dz, n, &beta, dx, k));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int m = xshape.row();
        int k = xshape.col();
        int b = yshape.batch;

        CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, m, b, &alpha, y, k, dz, m, &beta, dx, k));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        for (int b = 0; b < batch; ++b) {
            auto dxptr = dx + (b % xshape.batch) * xshape.batchSize();
            auto yptr  =  y + (b % yshape.batch) * yshape.batchSize();
            auto dzptr = dz + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, yptr, n, dzptr, n, &beta, dxptr, k));
        }
    }
}

template <>
void MatrixMultiplyGradXGPUImpl<double>( GPUDevice *device, 
                                        const double *x, 
                                        double *dx, 
                                        const Shape &xshape, 
                                        const double *y, 
                                        const Shape &yshape, 
                                        const double *z, 
                                        const double *dz, 
                                        const Shape &zshape) {
    double alpha = 1;
    double beta  = 1;

    if (1 == yhape.batch) {
        int b = xshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, b * m, n, &alpha, y, n, dz, n, &beta, dx, k));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int m = xshape.row();
        int k = xshape.col();
        int b = yshape.batch;

        CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, m, b, &alpha, y, k, dz, m, &beta, dx, k));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        for (int b = 0; b < batch; ++b) {
            auto dxptr = dx + (b % xshape.batch) * xshape.batchSize();
            auto yptr  =  y + (b % yshape.batch) * yshape.batchSize();
            auto dzptr = dz + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, yptr, n, dzptr, n, &beta, dxptr, k));
        }
    }
}

#ifdef HAVE_HALF
template <>
void MatrixMultiplyGradXGPUImpl<half>( GPUDevice *device, 
                                        const half *x, 
                                        half *dx, 
                                        const Shape &xshape, 
                                        const half *y, 
                                        const Shape &yshape, 
                                        const half *z, 
                                        const half *dz, 
                                        const Shape &zshape) {
    half alpha = 1.0;
    half beta  = 1.0;

    if (1 == yhape.batch) {
        int b = xshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, b * m, n, &alpha, y, n, dz, n, &beta, dx, k));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int m = xshape.row();
        int k = xshape.col();
        int b = yshape.batch;

        CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, k, m, b, &alpha, y, k, dz, m, &beta, dx, k));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        for (int b = 0; b < batch; ++b) {
            auto dxptr = dx + (b % xshape.batch) * xshape.batchSize();
            auto yptr  =  y + (b % yshape.batch) * yshape.batchSize();
            auto dzptr = dz + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, yptr, n, dzptr, n, &beta, dxptr, k));
        }
    }
}
#endif

void MatrixMultiplyGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    auto device = (GPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
    MatrixMultiplyGradXGPUImpl<float>(device, 
                                        x.data<float>(), 
                                        dx.data<float>(), 
                                        x.shape, 
                                        y.data<float>(), 
                                        y.shape, 
                                        z.data<float>(), 
                                        dz.data<float>(), 
                                        z.shape);
        break;
    case DType::Float64:
    MatrixMultiplyGradXGPUImpl<double>(device, 
                                    x.data<double>(), 
                                    dx.data<double>(), 
                                    x.shape, 
                                    y.data<double>(), 
                                    y.shape, 
                                    z.data<double>(), 
                                    dz.data<double>(), 
                                    z.shape);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
    MatrixMultiplyGradXGPUImpl<half>(device, 
                                    x.data<half>(), 
                                    dx.data<half>(), 
                                    x.shape, 
                                    y.data<half>(), 
                                    y.shape, 
                                    z.data<half>(), 
                                    dz.data<half>(), 
                                    z.shape);
        break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


/**gradient for Y*/
template <typename T>
void MatrixMultiplyGradYGPUImpl(GPUDevice *device, 
                                const T *x, 
                                const Shape &xshape, 
                                const T *y, 
                                T *dy,
                                const Shape &yshape, 
                                const T *z, 
                                const T *dz, 
                                const Shape &zshape) {
    DEEP8_RUNTIME_ERROR("the type in not support");
}

template <>
void MatrixMultiplyGradYGPUImpl<float>( GPUDevice *device, 
                                        const float *x, 
                                        const Shape &xshape, 
                                        const float *y, 
                                        float *dy,
                                        const Shape &yshape, 
                                        const float *z, 
                                        const float *dz, 
                                        const Shape &zshape) {
    float alpha = 1;
    float beta = 1;

    if (1 == yshape.batch) {
        int b = xshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m * b, &alpha, dz, n, x, k, &beta, dy, n));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int m = xshape.row();
        int k = xshape.col();
        int b = yshape.batch;

        CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, b, m, &alpha, x, k, dz, m, &beta, dy, k));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        for (int b = 0; b < batch; ++b) {
            auto  xptr =  x + (b % xshape.batch) * xshape.batchSize();
            auto dyptr = dy + (b % yshape.batch) * yshape.batchSize();
            auto dzptr = dz + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, dzptr, n, xptr, k, &beta, dyptr, n));
        }
    }                                        
}

template <>
void MatrixMultiplyGradYGPUImpl<double>( GPUDevice *device, 
                                        const double *x, 
                                        const Shape &xshape, 
                                        const double *y, 
                                        double *dy,
                                        const Shape &yshape, 
                                        const double *z, 
                                        const double *dz, 
                                        const Shape &zshape) {
    double alpha = 1;
    double beta = 1;

    if (1 == yshape.batch) {
        int b = xshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m * b, &alpha, dz, n, x, k, &beta, dy, n));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int m = xshape.row();
        int k = xshape.col();
        int b = yshape.batch;

        CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, b, m, &alpha, x, k, dz, m, &beta, dy, k));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        for (int b = 0; b < batch; ++b) {
            auto  xptr =  x + (b % xshape.batch) * xshape.batchSize();
            auto dyptr = dy + (b % yshape.batch) * yshape.batchSize();
            auto dzptr = dz + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, dzptr, n, xptr, k, &beta, dyptr, n));
        }
    }                                        
}

#ifdef HAVE_HALF
template <>
void MatrixMultiplyGradYGPUImpl<half>( GPUDevice *device, 
                                        const half *x, 
                                        const Shape &xshape, 
                                        const half *y, 
                                        half *dy,
                                        const Shape &yshape, 
                                        const half *z, 
                                        const half *dz, 
                                        const Shape &zshape) {
    half alpha = 1.0;
    half beta = 1.0;

    if (1 == yshape.batch) {
        int b = xshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m * b, &alpha, dz, n, x, k, &beta, dy, n));
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        int m = xshape.row();
        int k = xshape.col();
        int b = yshape.batch;

        CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, b, m, &alpha, x, k, dz, m, &beta, dy, k));
    } else {
        int batch = zshape.batch;
        int m = xshape.row();
        int k = xshape.col();
        int n = yshape.col();

        for (int b = 0; b < batch; ++b) {
            auto  xptr =  x + (b % xshape.batch) * xshape.batchSize();
            auto dyptr = dy + (b % yshape.batch) * yshape.batchSize();
            auto dzptr = dz + (b % zshape.batch) * zshape.batchSize();

            CUBLAS_CHECK(cublasHgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, dzptr, n, xptr, k, &beta, dyptr, n));
        }
    }                                        
}
#endif

void MatrixMultiplyGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto device = (GPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        MatrixMultiplyGradYGPUImpl<float>(device, 
                                            x.data<float>(), 
                                            x.shape, 
                                            y.data<float>(), 
                                            dy.data<float>(),
                                            y.shape, 
                                            z.data<float>(), 
                                            dz.data<float>(), 
                                            z.shape);
        break;
    case DType::Float64:
        MatrixMultiplyGradYGPUImpl<double>(device, 
                                        x.data<double>(), 
                                        x.shape, 
                                        y.data<double>(), 
                                        dy.data<double>(),
                                        y.shape, 
                                        z.data<double>(), 
                                        dz.data<double>(), 
                                        z.shape);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        MatrixMultiplyGradYGPUImpl<half>(device, 
                                        x.data<half>(), 
                                        x.shape, 
                                        y.data<half>(), 
                                        dy.data<half>(),
                                        y.shape, 
                                        z.data<half>(), 
                                        dz.data<half>(), 
                                        z.shape);
        break;
#endif
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


}
}