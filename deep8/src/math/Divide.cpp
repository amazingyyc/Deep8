#include "utils/ShapeUtils.h"
#include "math/Divide.h"

namespace Deep8 {
namespace Math {

/**
 * z = x / y 
 */
void Divide(const Tensor &x, const Tensor &y, Tensor &z) {
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
        DivideCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        DivideGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

/**
 * calculate grad(x) for z = x / y
 */
void DivideGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
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
        DivideGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        DivideGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

/**
 * calculate grad(y) for z = x / y
 */
void DivideGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
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
        DivideGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        DivideGradYGPU(x, y, dy, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}



template <typename T>
void DivideCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    auto xarray = enlargeShapeToMaxDim(xshape);
    auto yarray = enlargeShapeToMaxDim(yshape);
    auto zarray = enlargeShapeToMaxDim(zshape);

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int) xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int) yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, (int) zshape.size());

    if (xshape ==zshape && yshape ==zshape) {
        zvec.device(*eigenDevice) = xvec / yvec;
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
                xvec.reshape(xarray).broadcast(xbroad) / yvec.reshape(yarray).broadcast(ybroad);
    }
}

void DivideCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        DivideCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), z.shape);
        break;
    case DType::Float64:
        DivideCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

/**gradient for X*/
template <typename T, int diffCount> 
void DivideGradXCPUImpl(CPUDevice *device, 
                        T *dx, 
                        const Shape &xshape, 
                        T *y, 
                        const Shape &yshape, 
                        T *dz, 
                        const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    auto xarray = enlargeShapeToMaxDim(xshape);
    auto yarray = enlargeShapeToMaxDim(yshape);
    auto zarray = enlargeShapeToMaxDim(zshape);

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i <= MAX_TENSOR_DIMS; ++i) {
		if (xarray[i] != zarray[i]) {
			sumDims[j++] = i;
		}
	}

    auto ybroad = yarray;

	for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
		if (ybroad[i] < zarray[i]) {
			ybroad[i] = zarray[i];
		} else {
			ybroad[i] = 1;
		}
	}

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  yvec( y, (int)yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dzvec(dz, (int)zshape.size());

    dxvec.reshape(xarray).device(*eigenDevice) +=
		((dzvec.reshape(zarray)) / (yvec.reshape(yarray).broadcast(ybroad))).sum(sumDims).reshape(xarray);
}

void DivideGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    auto xarray = enlargeShapeToMaxDim(dx.shape);
    auto zarray = enlargeShapeToMaxDim(dz.shape);

    int diffCount = 0;

    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (xarray[i] != zarray[i]) {
            diffCount++;
        }
    }

    auto device = (CPUDevice*) dx.device();

    switch (x.type.id) {
    case DType::Float32:
        
        switch (diffCount) {
        case 0:
            DivideGradXCPUImpl<float, 0>(device, dx.data<float>(), dx.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
            break;
        case 1:
            DivideGradXCPUImpl<float, 1>(device, dx.data<float>(), dx.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
            break;
        case 2:
            DivideGradXCPUImpl<float, 2>(device, dx.data<float>(), dx.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
            break;
        case 3:
            DivideGradXCPUImpl<float, 3>(device, dx.data<float>(), dx.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
            break;
        case 4:
            DivideGradXCPUImpl<float, 4>(device, dx.data<float>(), dx.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
            break;
        case 5:
            DivideGradXCPUImpl<float, 5>(device, dx.data<float>(), dx.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
            break;
        default:
            DEEP8_RUNTIME_ERROR("the shape is error");
            break;
        }

        break;
    case DType::Float64:
        
        switch (diffCount) {
        case 0:
            DivideGradXCPUImpl<double, 0>(device, dx.data<double>(), dx.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
            break;
        case 1:
            DivideGradXCPUImpl<double, 1>(device, dx.data<double>(), dx.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
            break;
        case 2:
            DivideGradXCPUImpl<double, 2>(device, dx.data<double>(), dx.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
            break;
        case 3:
            DivideGradXCPUImpl<double, 3>(device, dx.data<double>(), dx.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
            break;
        case 4:
            DivideGradXCPUImpl<double, 4>(device, dx.data<double>(), dx.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
            break;
        case 5:
            DivideGradXCPUImpl<double, 5>(device, dx.data<double>(), dx.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
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

/**gradient for y*/
template <typename T, int diffCount>
void DivideGradYCPUImpl(CPUDevice *device, 
                        T *x, 
                        const Shape &xshape, 
                        T *y,
                        T *dy, 
                        const Shape &yshape, 
                        T *dz, 
                        const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    auto xarray = enlargeShapeToMaxDim(xshape);
    auto yarray = enlargeShapeToMaxDim(yshape);
    auto zarray = enlargeShapeToMaxDim(zshape);

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i <= MAX_TENSOR_DIMS; ++i) {
		if (yarray[i] != zarray[i]) {
			sumDims[j++] = i;
		}
	}

    auto xbroad = xarray;
    auto yboard = yarray;

	for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
		if (xbroad[i] < zarray[i]) {
			xbroad[i] = zarray[i];
		} else {
			xbroad[i] = 1;
		}

        if (yboard[i] < zarray[i]) {
			yboard[i] = zarray[i];
		} else {
			yboard[i] = 1;
		}
	}

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  yvec( y, (int)yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dzvec(dz, (int)zshape.size());

    dyvec.reshape(yarray).device(*eigenDevice) +=
        ((-dzvec.reshape(zarray) * xvec.reshape(xarray).broadcast(xbroad))
            / ((yvec.reshape(yarray).broadcast(yboard)) * (yvec.reshape(yarray).broadcast(yboard)))).sum(sumDims).reshape(yarray);
}

void DivideGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto yarray = enlargeShapeToMaxDim(dy.shape);
    auto zarray = enlargeShapeToMaxDim(dz.shape);

    int diffCount = 0;

    for (int i = 0; i <= MAX_TENSOR_DIMS; ++i) {
        if (yarray[i] != zarray[i]) {
            diffCount++;
        }
    }

    auto device = (CPUDevice*) dy.device();

    switch (x.type.id) {
    case DType::Float32:

        switch (diffCount) {
        case 0:
            DivideGradYCPUImpl<float, 0>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 1:
            DivideGradYCPUImpl<float, 1>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 2:
            DivideGradYCPUImpl<float, 2>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 3:
            DivideGradYCPUImpl<float, 3>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 4:
            DivideGradYCPUImpl<float, 4>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        case 5:
            DivideGradYCPUImpl<float, 5>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), dy.shape, dz.data<float>(), dz.shape);
            break;
        default:
            DEEP8_RUNTIME_ERROR("the shape is error");
            break;
        }

        break;
    case DType::Float64:

        switch (diffCount) {
        case 0:
            DivideGradYCPUImpl<double, 0>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 1:
            DivideGradYCPUImpl<double, 1>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 2:
            DivideGradYCPUImpl<double, 2>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 3:
            DivideGradYCPUImpl<double, 3>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 4:
            DivideGradYCPUImpl<double, 4>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
            break;
        case 5:
            DivideGradYCPUImpl<double, 5>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), dy.shape, dz.data<double>(), dz.shape);
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