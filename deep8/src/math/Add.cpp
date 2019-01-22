#include "utils/TensorUtils.h"
#include "math/Add.h"

namespace Deep8 {
namespace Math {

void Add(const Tensor &x, const Tensor &y, Tensor &z) {
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
        AddCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        AddGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

template <typename T>
void AddCPUImpl(CPUDevice *device,
                const T *x, Shape &xshape,
                const T *y, Shape &yshape,
                      T *z, Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    xarray = enlargeShapeToMaxDim(xshape);
    yarray = enlargeShapeToMaxDim(yshape);
    zarray = enlargeShapeToMaxDim(zshape);

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int) xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int) yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, (int) zshape.size());

    if (xshape == zshape && yshape == zshape) {
        zvec.device(*eigenDevice) = xvec + yvec;
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
                xvec.reshape(xarray).broadcast(xbroad) + yvec.reshape(yarray).broadcast(ybroad);
    }
}

void AddCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        AddCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), z.shape);
        break;
    case DType::Float64:
        AddCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


/**
 * calculate the grad for Add
 */
template <typename T, int diffCount>
void AddGradCPUImpl(CPUDevice *device, T *dxy, Shape &xyshape, const T *dz, Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxyvec(dxy, (int)xyshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dzvec(dz, (int)zshape.size());

    if (0 == diffCount) {
        dxyvec.device(*eigenDevice) += dzvec;
        return;
    }

    auto xyarray = enlargeShapeToMaxDim(xyshape);
    auto zarray = enlargeShapeToMaxDim(zshape);

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (xyarray[i] != zarray[i]) {
            sumDims[j++] = i;
        }
    }

    dxyvec.reshape(xyarray).device(*eigenDevice) += dzvec.reshape(zarray).sum(sumDims).reshape(xyarray);
}

/**
 * calculate grad(x) for z = x + y
 */
void AddGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
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
        AddGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        AddGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}


void AddGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
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
            AddGradCPUImpl<float, 0>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 1:
            AddGradCPUImpl<float, 1>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 2:
            AddGradCPUImpl<float, 2>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 3:
            AddGradCPUImpl<float, 3>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 4:
            AddGradCPUImpl<float, 4>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        case 5:
            AddGradCPUImpl<float, 5>(dx.device(), dx.data<float>(), dx.shape, dz.data<float>(), dz.shape);
            break;
        default:
            break;
        }

        break;
    case DType::Float64:
        
        switch (diffCout) {
        case 0:
            AddGradCPUImpl<double, 0>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 1:
            AddGradCPUImpl<double, 1>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 2:
            AddGradCPUImpl<double, 2>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 3:
            AddGradCPUImpl<double, 3>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 4:
            AddGradCPUImpl<double, 4>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        case 5:
            AddGradCPUImpl<double, 5>(dx.device(), dx.data<double>(), dx.shape, dz.data<double>(), dz.shape);
            break;
        default:
            break;
        }

        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

/**
 * calculate grad(y) for z = x + y
 */
void AddGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
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
        AddGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        AddGradYGPU(x, y, dy, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void AddGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto yarray = enlargeShapeToMaxDim(dy.shape);
    auto zarray = enlargeShapeToMaxDim(dz.shape);

    int diffCout = 0;

    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (yarray[i] != zarray[i]) {
            diffCout++;
        }
    }

    switch (x.type.id) {
    case DType::Float32:

        switch (diffCout) {
        case 0:
            AddGradCPUImpl<float, 0>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 1:
            AddGradCPUImpl<float, 1>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 2:
            AddGradCPUImpl<float, 2>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 3:
            AddGradCPUImpl<float, 3>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 4:
            AddGradCPUImpl<float, 4>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 5:
            AddGradCPUImpl<float, 5>(dy.device(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        default:
            break;
        }

        break;
    case DType::Float64:

        switch (diffCout) {
        case 0:
            AddGradCPUImpl<double, 0>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 1:
            AddGradCPUImpl<double, 1>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 2:
            AddGradCPUImpl<double, 2>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 3:
            AddGradCPUImpl<double, 3>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 4:
            AddGradCPUImpl<double, 4>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 5:
            AddGradCPUImpl<double, 5>(dy.device(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        default:
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