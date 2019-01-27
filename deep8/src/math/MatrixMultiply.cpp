#include "math/MatrixMultiply.h"

namespace Deep8 {
namespace Math {

template <typename T>
void MatrixMultiplyCPUImpl(const T *x, const Shape &xshape, const T *y, const Shape &yshape, T *z, const Shape &zshape) {
    if (1 == yshape.batch {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xmat(x, xshape.batch * xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ymat(y, yshape.row(), yshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> zmat(z, zshape.batch * zshape.row(), zshape.col());

        zmat.noalias() = xmat * ymat;
    } else if (1 == xshape.batch && 1 == y.col()) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xmat(x, xshape.row(), xshape.col());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ymat(y, yshape.batch, yshape.batchSize());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> zmat(y, zshape.batch, zshape.batchSize());

        zmat.noalias() = ymat * xmat.transpose();
    } else {
        for (size_t b = 0; b < zshape.batch, ++b) {
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

}
}