#include "math/MatrixMultiply.h"

namespace Deep8 {
namespace Math {

/**z = x * y*/
void MatrixMultiply(const Tensor &x, const Tensor &y, Tensor &z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType() && x.deviceType() == z.deviceType(), "the device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type == y.type && x.type == z.type, "the data type must be same");
    
    auto xshape = x.shape;
    auto yshape = y.shape;
    auto zshape = z.shape;

    DEEP8_ARGUMENT_CHECK(1 == xshape.batch || xshape.batch == zshape.batch, "the shape is error");
    DEEP8_ARGUMENT_CHECK(1 == yshape.batch || yshape.batch == zshape.batch, "the shape is error");
    DEEP8_ARGUMENT_CHECK((1 == xshape.nDims || 2 == xshape.nDims) &&
                         (1 == yshape.nDims || 2 == yshape.nDims) &&
                         (1 == zshape.nDims || 2 == zshape.nDims),
                         "the shape is error");
    DEEP8_ARGUMENT_CHECK(xshape.row() == zshape.row() && yshape.col() == zshape.col() && xshape.col() == yshape.row(), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        MatrixMultiplyCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        MatrixMultiplyGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

/**gradient for x*/
void MatrixMultiplyGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == z.deviceType() && x.deviceType() == dz.deviceType(), "the device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type == dx.type && x.type == y.type && x.type == z.type && x.type == dz.type, "the data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && z.shape == dz.shape, "the shape is error");

    auto xshape = x.shape;
    auto yshape = y.shape;
    auto zshape = z.shape;

    DEEP8_ARGUMENT_CHECK(1 == xshape.batch || xshape.batch == zshape.batch, "the shape is error");
    DEEP8_ARGUMENT_CHECK(1 == yshape.batch || yshape.batch == zshape.batch, "the shape is error");
    DEEP8_ARGUMENT_CHECK((1 == xshape.nDims || 2 == xshape.nDims) &&
                         (1 == yshape.nDims || 2 == yshape.nDims) &&
                         (1 == zshape.nDims || 2 == zshape.nDims),
                         "the shape is error");
    DEEP8_ARGUMENT_CHECK(xshape.row() == zshape.row() && yshape.col() == zshape.col() && xshape.col() == yshape.row(), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        MatrixMultiplyGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        MatrixMultiplyGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}


/**gradient for y*/
void MatrixMultiplyGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType() && x.deviceType() == z.deviceType() && x.deviceType() == dz.deviceType(), "the device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type == y.type && x.type == dy.type && x.type == z.type && x.type == dz.type, "the data type must be same");
    DEEP8_ARGUMENT_CHECK(y.shape == dy.shape && z.shape == dz.shape, "the shape is error");

    auto xshape = x.shape;
    auto yshape = y.shape;
    auto zshape = z.shape;

    DEEP8_ARGUMENT_CHECK(1 == xshape.batch || xshape.batch == zshape.batch, "the shape is error");
    DEEP8_ARGUMENT_CHECK(1 == yshape.batch || yshape.batch == zshape.batch, "the shape is error");
    DEEP8_ARGUMENT_CHECK((1 == xshape.nDims || 2 == xshape.nDims) &&
                         (1 == yshape.nDims || 2 == yshape.nDims) &&
                         (1 == zshape.nDims || 2 == zshape.nDims),
                         "the shape is error");
    DEEP8_ARGUMENT_CHECK(xshape.row() == zshape.row() && yshape.col() == zshape.col() && xshape.col() == yshape.row(), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        MatrixMultiplyGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        MatrixMultiplyGradYGPU(x, y, dy, z, dz);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

template <typename T>
void MatrixMultiplyCPUImpl(T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
    if (1 == yshape.batch) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xmat(x, xshape.batch * xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ymat(y, yshape.row(), yshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> zmat(z, zshape.batch * zshape.row(), zshape.col());

        zmat.noalias() = xmat * ymat;
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xmat(x, xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ymat(y, yshape.batch, yshape.batchSize());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> zmat(z, zshape.batch, zshape.batchSize());

        zmat.noalias() = ymat * xmat.transpose();
    } else {
        for (size_t b = 0; b < zshape.batch; ++b) {
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xmat(x + (b % xshape.batch) * xshape.batchSize(), xshape.row(), xshape.col());
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ymat(y + (b % yshape.batch) * yshape.batchSize(), yshape.row(), yshape.col());
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> zmat(z + (b % zshape.batch) * zshape.batchSize(), zshape.row(), zshape.col());

            zmat.noalias() = xmat * ymat;
        }
    }
}

void MatrixMultiplyCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    switch (x.type.id) {
    case DType::Float32:
        MatrixMultiplyCPUImpl<float>(x.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), z.shape);
        break;
    case DType::Float64:
        MatrixMultiplyCPUImpl<double>(x.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


template <typename T>
void MatrixMultiplyGradXCPUImpl(T *x, 
                                T *dx, 
                                const Shape &xshape, 
                                T *y, 
                                const Shape &yshape,
                                T *z, 
                                T *dz, 
                                const Shape &zshape) {
    if (1 == yshape.batch) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dxmat(dx, xshape.batch * xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  ymat( y, yshape.row(), yshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dzmat(dz, zshape.batch * zshape.row(), zshape.col());

        dxmat.noalias() += dzmat * ymat.transpose();
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dxmat(dx, xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  ymat(y,  yshape.batch, yshape.batchSize());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dzmat(dz, zshape.batch, zshape.batchSize());

        dxmat.noalias() += dzmat.transpose() * ymat;
    } else {
        for (size_t b = 0; b < zshape.batch; ++b) {
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dxmat(dx + (b % xshape.batch) * xshape.batchSize(), xshape.row(), xshape.col());
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  ymat(y  + (b % yshape.batch) * yshape.batchSize(), yshape.row(), yshape.col());
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dzmat(dz + (b % zshape.batch) * zshape.batchSize(), zshape.row(), zshape.col());

            dxmat.noalias() += dzmat * ymat.transpose();
        }
    }
}

void MatrixMultiplyGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    switch (x.type.id) {
    case DType::Float32:
        MatrixMultiplyGradXCPUImpl<float>(x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), dz.data<float>(), z.shape);
        break;
    case DType::Float64:
        MatrixMultiplyGradXCPUImpl<double>(x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), dz.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


template <typename T>
void MatrixMultiplyGradYCPUImpl(T *x, 
                                const Shape &xshape, 
                                T *y, 
                                T *dy,
                                const Shape &yshape,
                                T *z, 
                                T *dz, 
                                const Shape &zshape) {
    if (1 == yshape.batch) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  xmat(x, xshape.batch * xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dymat(dy, yshape.row(), yshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dzmat(dz, zshape.batch * zshape.row(), zshape.col());

        dymat.noalias() += xmat.transpose() * dzmat;
    } else if (1 == xshape.batch && 1 == yshape.col()) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  xmat(x, xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dymat(dy, yshape.batch, yshape.batchSize());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dzmat(dz, zshape.batch, zshape.batchSize());

        dymat.noalias() += dzmat * xmat;
    } else {
        for (size_t b = 0; b < zshape.batch; ++b) {
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  xmat(x  + (b % xshape.batch) * xshape.batchSize(), xshape.row(), xshape.col());
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dymat(dy + (b % yshape.batch) * yshape.batchSize(), yshape.row(), yshape.col());
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dzmat(dz + (b % zshape.batch) * zshape.batchSize(), zshape.row(), zshape.col());
            
            dymat.noalias() += xmat.transpose() * dzmat;
        }
    }
}

void MatrixMultiplyGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    switch (x.type.id) {
    case DType::Float32:
        MatrixMultiplyGradYCPUImpl<float>(x.data<float>(), x.shape, y.data<float>(), dy.data<float>(),  y.shape, z.data<float>(), dz.data<float>(), z.shape);
        break;
    case DType::Float64:
        MatrixMultiplyGradYCPUImpl<double>(x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, z.data<double>(), dz.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

}
}