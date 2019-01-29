#include "math/Minus.h"

namespace Deep8 {
namespace Math {

/**z = x - y*/
void Minus(const Tensor &x, const Tensor &y, Tensor &z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType() && x.deviceType()  == z.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == y.type && x.type  == z.type, "the param data type must be same");

    auto xarray = enlargeShapeToMaxDim(x.shape);
    auto yarray = enlargeShapeToMaxDim(y.shape);
    auto zarray = enlargeShapeToMaxDim(z.shape);
    
    /**check shape*/
    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        DEEP8_ARGUMENT_CHECK(1 == xarray[0] || xarray[0] == zarray[0], "the shape is error");
        DEEP8_ARGUMENT_CHECK(1 == yarray[0] || yarray[0] == zarray[0], "the shape is error");
    }

    if (DeviceType::CPU == x.deviceType()) {
        MinusCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        MinusGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}


/**gradient for x*/
void MinusGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && 
                         x.deviceType() ==  y.deviceType() &&
                         x.deviceType() ==  z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.type == dx.type &&
                         x.type ==  y.type &&
                         x.type ==  z.type &&
                         x.type == dz.type, "the param data type must be same");

    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape, "the x shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape, "the z shape is error");

    auto xarray = enlargeShapeToMaxDim(x.shape);
    auto yarray = enlargeShapeToMaxDim(y.shape);
    auto zarray = enlargeShapeToMaxDim(z.shape);

    /**check shape*/
    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        DEEP8_ARGUMENT_CHECK(1 == xarray[0] || xarray[0] == zarray[0], "the shape is error");
        DEEP8_ARGUMENT_CHECK(1 == yarray[0] || yarray[0] == zarray[0], "the shape is error");
    }

    if (DeviceType::CPU == x.deviceType()) {
        MinusGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        MinusGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

/**gradient for y*/
void MinusGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() ==  y.deviceType() &&
                         x.deviceType() == dy.deviceType() &&
                         x.deviceType() ==  z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.type ==  y.type &&
                         x.type == dy.type &&
                         x.type ==  z.type &&
                         x.type == dz.type, "the param data type must be same");

    DEEP8_ARGUMENT_CHECK(y.shape == dy.shape, "the y shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape, "the z shape is error");

    auto xarray = enlargeShapeToMaxDim(x.shape);
    auto yarray = enlargeShapeToMaxDim(y.shape);
    auto zarray = enlargeShapeToMaxDim(z.shape);

    /**check shape*/
    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        DEEP8_ARGUMENT_CHECK(1 == xarray[0] || xarray[0] == zarray[0], "the shape is error");
        DEEP8_ARGUMENT_CHECK(1 == yarray[0] || yarray[0] == zarray[0], "the shape is error");
    }

    if (DeviceType::CPU == x.deviceType()) {
        MinusGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        MinusGradYGPU(x, y, dy, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}


/**z = x - y*/
template <typename T>
void MinusCPUImpl(CPUDevice *device, const T *x, const Shape &xshape, const T *y, const Shape &yshape, T *z, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    xarray = enlargeShapeToMaxDim(xshape);
    yarray = enlargeShapeToMaxDim(yshape);
    zarray = enlargeShapeToMaxDim(zshape);

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int) xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int) yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, (int) zshape.size());

    if (xshape == zshape && yshape == zshape) {
        zvec.device(*eigenDevice) = xvec - yvec;
    } else {
        auto xbroad = xarray;
        auto ybroad = yarray;

        for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
            if (xbroad[i] < zarray[i]) {
                xbroad[i] = zarray[i];
            } else {
                xbroad[i] = 1;
            }

            if (ybroad[i] < zarray[i]) {
                ybroad[i] = zarray[i];
            } else {
                ybroad[i] = 1;
            }
        }

        zvec.reshape(zarray).device(*eigenDevice) = 
                xvec.reshape(xarray).broadcast(xbroad) - yvec.reshape(yarray).broadcast(ybroad);
    }
}

void MinusCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        MinusCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), z.shape);
        break;
    case DType::Float64:
        MinusCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


template <typename T, int diffCount>
void MinusGradXCPUImpl( CPUDevice *device, 
                        T *dx, 
                        const Shape &xshape, 
                        const T *dz, 
                        const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dzvec(dz, (int)zshape.size());

    if (0 == diffCount) {
        dxvec.device(*eigenDevice) += dzvec;
        return;
    }

    auto xarray = enlargeShapeToMaxDim(xshape);
    auto zarray = enlargeShapeToMaxDim(zshape);

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (xarray[i] != zarray[i]) {
            sumDims[j++] = i;
        }
    }

    dxvec.reshape(xarray).device(*eigenDevice) += dzvec.reshape(zarray).sum(sumDims).reshape(xarray);
}

void MinusGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    auto xarray = enlargeShapeToMaxDim(dx.shape);
    auto zarray = enlargeShapeToMaxDim(dz.shape);

    int diffCout = 0;

    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (xarray[i] != zarray[i]) {
            diffCout++;
        }
    }

    switch (x.type.id) {
    case DType::Float32:
        
        switch (diffCout) {
        case 0:
            MinusGradXCPUImpl<float, 0>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 1:
            MinusGradXCPUImpl<float, 1>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 2:
            MinusGradXCPUImpl<float, 2>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 3:
            MinusGradXCPUImpl<float, 3>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 4:
            MinusGradXCPUImpl<float, 4>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 5:
            MinusGradXCPUImpl<float, 5>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        default:
            DEEP8_RUNTIME_ERROR("the shape is error");
            break;
        }

        break;
    case DType::Float64:
        
        switch (diffCout) {
        case 0:
            MinusGradXCPUImpl<double, 0>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 1:
            MinusGradXCPUImpl<double, 1>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 2:
            MinusGradXCPUImpl<double, 2>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 3:
            MinusGradXCPUImpl<double, 3>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 4:
            MinusGradXCPUImpl<double, 4>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 5:
            MinusGradXCPUImpl<double, 5>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        default:
            DEEP8_RUNTIME_ERROR("the shape is error");
            break;
        }

        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T, int diffCount>
void MinusGradYCPUImpl( CPUDevice *device, 
                        T *dy, 
                        const Shape &yshape, 
                        const T *dz, 
                        const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dzvec(dz, (int)zshape.size());

    if (0 == diffCount) {
        dyvec.device(*eigenDevice) -= dzvec;
        return;
    }

    auto yarray = enlargeShapeToMaxDim(yshape);
    auto zarray = enlargeShapeToMaxDim(zshape);

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (yarray[i] != zarray[i]) {
            sumDims[j++] = i;
        }
    }

    dyvec.reshape(xarray).device(*eigenDevice) -= dzvec.reshape(zarray).sum(sumDims).reshape(yarray);
}

void MinusGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto yarray = enlargeShapeToMaxDim(dy.shape);
    auto zarray = enlargeShapeToMaxDim(dz.shape);

    int diffCount = 0;

    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (yarray[i] != zarray[i]) {
            diffCount++;
        }
    }

    switch (x.type.id) {
    case DType::Float32:

        switch (diffCount) {
        case 0:
            MinusGradYCPUImpl<float, 0>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 1:
            MinusGradYCPUImpl<float, 1>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 2:
            MinusGradYCPUImpl<float, 2>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 3:
            MinusGradYCPUImpl<float, 3>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 4:
            MinusGradYCPUImpl<float, 4>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 5:
            MinusGradYCPUImpl<float, 5>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        default:
            DEEP8_RUNTIME_ERROR("the shape is error");
            break;
        }

        break;
    case DType::Float64:

        switch (diffCount) {
        case 0:
            MinusGradYCPUImpl<double, 0>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 1:
            MinusGradYCPUImpl<double, 1>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 2:
            MinusGradYCPUImpl<double, 2>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 3:
            MinusGradYCPUImpl<double, 3>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 4:
            MinusGradYCPUImpl<double, 4>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 5:
            MinusGradYCPUImpl<double, 5>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        default:
            DEEP8_RUNTIME_ERROR("the shape is error");
            break;
        }

        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}




}
}