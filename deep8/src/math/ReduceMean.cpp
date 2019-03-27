#include "utils/ShapeUtils.h"
#include "math/ReduceMean.h"

namespace Deep8 {
namespace Math {

void ReduceMean(const Tensor &x, Tensor &y, const std::vector<int> &axis, bool keepDims) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType, "the param data type must be same");

    auto xshape = convertShapeToVector(x.shape);
    int rank = xshape.size();

    std::vector<bool> reduceAxis(rank);

    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = true;
        }
    } else {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = false;
        }

        for (int i = 0; i < axis.size(); ++i) {
            DEEP8_ARGUMENT_CHECK(-rank <= axis[i] && axis[i] < rank, "the reduce dim is error");

            reduceAxis[(axis[i] + rank) % rank] = true;
        }
    }

    DEEP8_ARGUMENT_CHECK(reduceAxis.size() >= 2, "the shape is error");

    size_t ybatch = reduceAxis[0] ? 1 : xshape[0];
    std::vector<size_t> ylist;

    for(int i = 1; i < rank; ++i) {
        if (reduceAxis[i]) {
            if (keepDims) {
                ylist.emplace_back(1);
            }
        } else {
            ylist.emplace_back(xshape[i]);
        }
    }

    if (ylist.empty()) {
        ylist.emplace_back(1);
    }

    Shape yshape(ybatch, ylist);

    DEEP8_ARGUMENT_CHECK(yshape == y.shape, "the y shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        ReduceMeanCPU(x, y, axis, keepDims);
    } else {
#ifdef HAVE_CUDA
        ReduceMeanGPU(x, y, axis, keepDims);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void ReduceMeanGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, const std::vector<int> &axis, bool keepDims) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == dx.elementType  && x.elementType == y.elementType && x.elementType  == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && y.shape == dy.shape, "the x/dx or y/dy shape must be same");

    auto xshape = convertShapeToVector(x.shape);
    int rank = xshape.size();

    std::vector<bool> reduceAxis(rank);

    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = true;
        }
    } else {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = false;
        }

        for (int i = 0; i < axis.size(); ++i) {
            DEEP8_ARGUMENT_CHECK(-rank <= axis[i] && axis[i] < rank, "the reduce dim is error");

            reduceAxis[(axis[i] + rank) % rank] = true;
        }
    }

    DEEP8_ARGUMENT_CHECK(reduceAxis.size() >= 2, "the shape is error");

    size_t ybatch = reduceAxis[0] ? 1 : xshape[0];
    std::vector<size_t> ylist;

    for(int i = 1; i < rank; ++i) {
        if (reduceAxis[i]) {
            if (keepDims) {
                ylist.emplace_back(1);
            }
        } else {
            ylist.emplace_back(xshape[i]);
        }
    }

    if (ylist.empty()) {
        ylist.emplace_back(1);
    }

    Shape yshape(ybatch, ylist);

    DEEP8_ARGUMENT_CHECK(yshape == y.shape, "the y shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        ReduceMeanGradCPU(x, dx, y, dy, axis, keepDims);
    } else {
#ifdef HAVE_CUDA
        ReduceMeanGradGPU(x, dx, y, dy, axis, keepDims);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T, int NumDims, int ReduceCount>
void ReduceMeanCPUEigenImpl(CPUDevice *device, T *x, std::vector<int> &xarray, T *y, std::vector<int> &yarray) {
    auto eigenDevice = device->eigenDevice;

    Eigen::array<int, NumDims> xshape;
    Eigen::array<int, NumDims> yshape;

    for (int i = 0; i < NumDims; ++i) {
        xshape[i] = xarray[i];
        yshape[i] = yarray[i];
    }

    Eigen::array<int, ReduceCount> reduceDims;

    for (int i = 0, j = 0; i < NumDims; ++i) {
        if (xshape[i] != yshape[i]) {
            reduceDims[j++] = i;
        }
    }

    int xsize = 1;
    int ysize = 1;

    for (int i = 0; i < NumDims; ++i) {
        xsize *= xshape[i];
        ysize *= yshape[i];
    }

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, xsize);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, ysize);

    yvec.reshape(yshape).device(*eigenDevice) = xvec.reshape(xshape).mean(reduceDims).reshape(yshape);
}

template <typename T>
void ReduceMeanCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape, const std::vector<int> &axis) {
    auto xarray = convertShapeToVector(xshape);
    int rank = xarray.size();

    std::vector<bool> reduceAxis(rank);

    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = true;
        }
    } else {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = false;
        }

        for (int i = 0; i < axis.size(); ++i) {
            DEEP8_ARGUMENT_CHECK(-rank <= axis[i] && axis[i] < rank, "the reduce dims is error");

            reduceAxis[(axis[i] + rank) % rank] = true;
        }
    }

    auto yarray = xarray;

    for (int i = 0; i < rank; ++i) {
        if (reduceAxis[i]) {
            yarray[i] = 1;
        } else {
            yarray[i] = xarray[i];
        }
    }

    int NumDims = rank;
    int ReduceCount = 0;

    for (int i = 0; i < rank; ++i) {
        if (yarray[i] != xarray[i]) {
            ReduceCount++;
        }
    }

    if (1 == NumDims) {
        if (0 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 1, 0>(device, x, xarray, y, yarray);
        } else if (1 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 1, 1>(device, x, xarray, y, yarray);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    } else if (2 == NumDims) {
        if (0 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 2, 0>(device, x, xarray, y, yarray);
        } else if (1 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 2, 1>(device, x, xarray, y, yarray);
        } else if (2 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 2, 2>(device, x, xarray, y, yarray);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    } else if (3 == NumDims) {
        if (0 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 3, 0>(device, x, xarray, y, yarray);
        } else if (1 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 3, 1>(device, x, xarray, y, yarray);
        } else if (2 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 3, 2>(device, x, xarray, y, yarray);
        } else if (3 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 3, 3>(device, x, xarray, y, yarray);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    } else if (4 == NumDims) {
        if (0 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 4, 0>(device, x, xarray, y, yarray);
        } else if (1 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 4, 1>(device, x, xarray, y, yarray);
        } else if (2 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 4, 2>(device, x, xarray, y, yarray);
        } else if (3 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 4, 3>(device, x, xarray, y, yarray);
        } else if (4 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 4, 4>(device, x, xarray, y, yarray);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    } else if (5 == NumDims) {
        if (0 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 5, 0>(device, x, xarray, y, yarray);
        } else if (1 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 5, 1>(device, x, xarray, y, yarray);
        } else if (2 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 5, 2>(device, x, xarray, y, yarray);
        } else if (3 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 5, 3>(device, x, xarray, y, yarray);
        } else if (4 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 5, 4>(device, x, xarray, y, yarray);
        } else if (5 == ReduceCount) {
            ReduceMeanCPUEigenImpl<T, 5, 5>(device, x, xarray, y, yarray);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    } else {
        DEEP8_RUNTIME_ERROR("the shape is error");
    }
}

void ReduceMeanCPU(const Tensor &x, Tensor &y, const std::vector<int> &axis, bool keepDims) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        ReduceMeanCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, axis);
        break;
    case DType::Float64:
        ReduceMeanCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, axis);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T, int NumDims>
void ReduceMeanGradCPUEigenImpl(CPUDevice*device, T *x, T *dx, std::vector<int> &xarray, T *y, T *dy, std::vector<int> &yarray) {
    auto eigenDevice = device->eigenDevice;

    Eigen::array<int, NumDims> xshape;
    Eigen::array<int, NumDims> yshape;

    for (int i = 0; i < NumDims; ++i) {
        xshape[i] = xarray[i];
        yshape[i] = yarray[i];
    }

    Eigen::array<int, NumDims> broadDims;

    int ratio = 1;

    for (int i = 0, j = 0; i < NumDims; ++i) {
        if (xshape[i] != yshape[i]) {
            broadDims[i] = xshape[i];

            ratio *= xshape[i];
        } else {
            broadDims[i] = 1;
        }
    }

    int xsize = 1;
    int ysize = 1;
    
    for (int i = 0; i < NumDims; ++i) {
        xsize *= xshape[i];
        ysize *= yshape[i];
    }

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, xsize);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, ysize);

    dxvec.reshape(xshape).device(*eigenDevice) += dyvec.reshape(yshape).broadcast(broadDims) / T(ratio);
}

template <typename T>
void ReduceMeanGradCPUImpl(CPUDevice*device, T *x, T *dx, const Shape &xshape, T *y, T *dy, const Shape &yshape, const std::vector<int> &axis) {
    auto xarray = convertShapeToVector(xshape);
    
    int rank = xarray.size();

    std::vector<bool> reduceAxis(rank);

    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = true;
        }
    } else {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = false;
        }

        for (int i = 0; i < axis.size(); ++i) {
            DEEP8_ARGUMENT_CHECK(-rank <= axis[i] && axis[i] < rank, "the reduce dims is error");

            reduceAxis[(axis[i] + rank) % rank] = true;
        }
    }

    auto yarray = xarray;

    for (int i = 0; i < rank; ++i) {
        if (reduceAxis[i]) {
            yarray[i] = 1;
        } else {
            yarray[i] = xarray[i];
        }
    }

    int NumDims = rank;

    if (1 == NumDims) {
        ReduceMeanGradCPUEigenImpl<T, 1>(device, x, dx, xarray, y, dy, yarray);
    } else if (2 == NumDims) {
        ReduceMeanGradCPUEigenImpl<T, 2>(device, x, dx, xarray, y, dy, yarray);
    } else if (3 == NumDims) {
        ReduceMeanGradCPUEigenImpl<T, 3>(device, x, dx, xarray, y, dy, yarray);
    } else if (4 == NumDims) {
        ReduceMeanGradCPUEigenImpl<T, 4>(device, x, dx, xarray, y, dy, yarray);
    } else if (5 == NumDims) {
        ReduceMeanGradCPUEigenImpl<T, 5>(device, x, dx, xarray, y, dy, yarray);
    } else {
        DEEP8_RUNTIME_ERROR("the shape is error");
    }
}

void ReduceMeanGradCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy, const std::vector<int>& axis, bool keepDims) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        ReduceMeanGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape, axis);
        break;
    case DType::Float64:
        ReduceMeanGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, axis);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}
}