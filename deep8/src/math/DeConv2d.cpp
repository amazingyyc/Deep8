#include "math/DeConv2d.h"

namespace Deep8 {
namespace Math {

void DeConv2d(  const Tensor &x, 
                const Tensor &y, 
                Tensor &z,
                void *zmat,
                bool convered,
                int strideY,
                int strideX) {
    DEEP8_ARGUMENT_CHECK(x.type == y.type && x.type == z.type, "the data type must be same");
    DEEP8_ARGUMENT_CHECK(strideY >= 1 && strideX >= 1 , "the stride must >= 1");
    DEEP8_ARGUMENT_CHECK(3 == x.nDims() && 4 == y.nDims() && 3 == z.nDims(), "the shape is error");

    auto inputBatch   = (int)x.batch();
    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto outputChannel = (int)y.dim(0);
    auto filterHeight  = (int)y.dim(1);
    auto filterWidth   = (int)y.dim(2);
    
    int outputHeight;
    int outputWidth;

    if (!convered) {
        outputHeight = (inputHeight - 1) * strideY + filterHeight;
        outputWidth  = (inputWidth  - 1) * strideX + filterWidth;
    } else {
        outputHeight = (inputHeight - 1) * strideY + 1 - strideY + filterHeight;
        outputWidth  = (inputWidth  - 1) * strideX + 1 - strideX + filterWidth;
    }

    DEEP8_ARGUMENT_CHECK(inputBatch   == (int)z.batch() && 
                    outputHeight  == (int)z.dim(0) &&
                    outputWidth   == (int)z.dim(1) &&
                    outputChannel == (int)z.dim(2), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        DeConv2dCPU(x, y, z, convered, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        DeConv2dGPU(x, y, z, zmat, convered, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}   

void DeConv2dGradX( const Tensor& x, 
                    Tensor& dx,
                    const Tensor& y,
                    const Tensor& z, 
                    const Tensor& dz,
                    void *dzmat,
                    bool convered,
                    int strideY,
                    int strideX) {
    DEEP8_ARGUMENT_CHECK(x.type == dx.type && x.type == y.type && x.type == z.type && x.type == dz.type, "the data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && z.shape == dz.shape, "the shape is error");
    DEEP8_ARGUMENT_CHECK(strideY >= 1 && strideX >= 1, "the stride must >= 1");
    DEEP8_ARGUMENT_CHECK(3 == x.nDims() && 4 == y.nDims() && 3 == z.nDims(), "the shape is error");

    auto inputBatch   = (int)x.batch();
    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto outputChannel = (int)y.dim(0);
    auto filterHeight  = (int)y.dim(1);
    auto filterWidth   = (int)y.dim(2);
    
    int outputHeight;
    int outputWidth;

    if (!convered) {
        outputHeight = (inputHeight - 1) * strideY + filterHeight;
        outputWidth  = (inputWidth  - 1) * strideX + filterWidth;
    } else {
        outputHeight = (inputHeight - 1) * strideY + 1 - strideY + filterHeight;
        outputWidth  = (inputWidth  - 1) * strideX + 1 - strideX + filterWidth;
    }

    DEEP8_ARGUMENT_CHECK(inputBatch   == (int)z.batch() && 
                    outputHeight  == (int)z.dim(0) &&
                    outputWidth   == (int)z.dim(1) &&
                    outputChannel == (int)z.dim(2), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        DeConv2dGradXCPU(x, dx, y, z, dz, convered, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        DeConv2dGradXGPU(x, dx, y, z, dz, dzmat, convered, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void DeConv2dGradY( const Tensor &x,
                    const Tensor &y, 
                    Tensor &dy,
                    const Tensor &z, 
                    const Tensor& dz,
                    void *dzmat,
                    bool convered,
                    int strideY,
                    int strideX) {
    DEEP8_ARGUMENT_CHECK(x.type == y.type && x.type == dy.type && x.type == z.type && x.type == dz.type, "the data type must be same");
    DEEP8_ARGUMENT_CHECK(y.shape == dy.shape && z.shape == dz.shape, "the shape is error");
    DEEP8_ARGUMENT_CHECK(strideY >= 1 && strideX >= 1, "the stride must >= 1");
    DEEP8_ARGUMENT_CHECK(3 == x.nDims() && 4 == y.nDims() && 3 == z.nDims(), "the shape is error");

    int outputHeight;
    int outputWidth;

    if (!convered) {
        outputHeight = (inputHeight - 1) * strideY + filterHeight;
        outputWidth  = (inputWidth  - 1) * strideX + filterWidth;
    } else {
        outputHeight = (inputHeight - 1) * strideY + 1 - strideY + filterHeight;
        outputWidth  = (inputWidth  - 1) * strideX + 1 - strideX + filterWidth;
    }

    DEEP8_ARGUMENT_CHECK(inputBatch   == (int)z.batch() && 
                    outputHeight  == (int)z.dim(0) &&
                    outputWidth   == (int)z.dim(1) &&
                    outputChannel == (int)z.dim(2), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        DeConv2dGradYCPU(x, y, dy, z, dz, convered, strideY, strideX);
    } else {
#ifdef HAVE_CUDA
        DeConv2dGradYGPU(x, y, dy, z, dz, dzmat, convered, strideY, strideX);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void DeConv2dCPUImpl(   CPUDevice *device, 
                        const T *x, 
                        const Shape &xshape, 
                        const T *y,
                        const Shape &yshape, 
                        T *z, 
                        const Shape &zshape,
                        bool convered,
                        int strideY,
                        int strideX) {
    auto eigenDevice = device->eigenDevice;

    auto batch = (int) xshape.batch;

    auto inputHeight  = (int) xshape.dim(0);
    auto inputWidth   = (int) xshape.dim(1);
    auto inputChannel = (int) xshape.dim(2);

    auto outputHeight  = (int) zshape.dim(0);
    auto outputWidth   = (int) zshape.dim(1);
    auto outputChannel = (int) zshape.dim(2);

    auto filterHeight = (int) yshape.dim(1);
    auto filterWidth  = (int) yshape.dim(2);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            xvec(x, batch, inputHeight, inputWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            yvec(y, outputChannel, filterHeight, filterWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            zvec(z, batch, outputHeight, outputWidth, outputChannel);

    Eigen::DSizes<int, 2> preContractDims;
    preContractDims[0] = batch * outputHeight * outputWidth;
    preContractDims[1] = filterHeight * filterWidth * inputChannel;

    Eigen::DSizes<int, 2> kernelDims;
    kernelDims[0] = outputChannel;
    kernelDims[1] = filterHeight * filterWidth * inputChannel;

    Eigen::DSizes<int, 2> filterShuffle;
    filterShuffle[0] = 1;
    filterShuffle[1] = 0;

    Eigen::array<Eigen::IndexPair<int>, 1> contractDims;
    contractDims[0] = Eigen::IndexPair<int>(1, 0);

    int padH = std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2);
    int padW = std::max<int>(0, outputWidth  + filterWidth  - (inputWidth  - 1) * strideX - 2);

    int padTop = padH / 2;
    int padBottom = padH - padTop;
    int padLeft = padW / 2;
    int padRight = padW - padLeft;

    zvec.device(*eigenDevice) = xvec.extract_image_patches(filterWidth, filterHeight, 1, 1, 1, 1, strideX, strideY, padTop, padBottom, padLeft, padRight, 0)
            .reshape(preContractDims)
            .contract(yvec.reshape(kernelDims).shuffle(filterShuffle), contractDims)
            .reshape(zvec.dimensions());
}

void DeConv2dCPU(const Tensor &x, const Tensor &y, Tensor &z, bool convered, int strideY, int strideX) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        DeConv2dCPUImpl<float>( device, 
                                x.data<float>(), 
                                x.shape, 
                                y.data<float>(), 
                                y.shape, 
                                z.data<float>(), 
                                z.shape,
                                convered,
                                strideY,
                                strideX);
        break;
    case DType::Float64:
        DeConv2dCPUImpl<double>( device, 
                                x.data<double>(), 
                                x.shape, 
                                y.data<double>(), 
                                y.shape, 
                                z.data<double>(), 
                                z.shape,
                                convered,
                                strideY,
                                strideX);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
void DeConv2dGradXCPUImpl(  CPUDevice *device,
                            const T *x,
                            T *dx,
                            const Shape &xshape,
                            const T *y,
                            const Shape &yahspe,
                            const T *z,
                            const T *da,
                            const Shape &zshape,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto eigenDevice = device->eigenDevice;

    auto batch = (int) xshape.batch;

    auto inputHeight  = (int) xshape.dim(0);
    auto inputWidth   = (int) xshape.dim(1);
    auto inputChannel = (int) xshape.dim(2);

    auto outputHeight  = (int) zshape.dim(0);
    auto outputWidth   = (int) zshape.dim(1);
    auto outputChannel = (int) zshape.dim(2);

    auto filterHeight = (int) yshape.dim(1);
    auto filterWidth  = (int) yshape.dim(2);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            dzvec(dz, batch, outputHeight, outputWidth, outputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            dxvec(dx, batch, inputHeight, inputWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            yvec(y, outputChannel, filterHeight, filterWidth, inputChannel);

    Eigen::DSizes<int, 2> preContractDims;
    preContractDims[0] = batch * inputHeight * inputWidth;
    preContractDims[1] = filterHeight * filterWidth * outputChannel;

    Eigen::internal::conditional<false, Eigen::array<bool, 4>, Eigen::array<bool, 4>>::type filterReverse;
    filterReverse[0] = false;
    filterReverse[1] = true;
    filterReverse[2] = true;
    filterReverse[3] = false;

    Eigen::DSizes<int, 4> filterShuffle;
    filterShuffle[0] = 1;
    filterShuffle[1] = 2;
    filterShuffle[2] = 0;
    filterShuffle[3] = 3;

    Eigen::DSizes<int, 2> filterDim;
    filterDim[0] = filterHeight * filterWidth * outputChannel;
    filterDim[1] = inputChannel;

    Eigen::array<Eigen::IndexPair<int>, 1> contractDims;
    contractDims[0] = Eigen::IndexPair<int>(1, 0);

    auto forwardPadTop = std::max<int>(0, (outputHeight + filterHeight - (inputHeight - 1) * strideY - 2) / 2);
    auto forwardPadBottom = std::max<int>(0, (outputWidth + filterWidth - (inputWidth - 1) * strideX - 2) / 2);

    auto padTop  = filterHeight - 1 - forwardPadTop;
    auto padLeft = filterWidth  - 1 - forwardPadBottom;
    auto padBottom = (inputHeight - 1) * strideY + filterHeight - outputHeight - padTop;
    auto padRight  = (inputWidth - 1) * strideX + filterWidth - outputWidth - padLeft;

    dxvec.device(*eigenDevice) += dzvec.extract_image_patches(filterWidth, filterHeight, strideX, strideY, 1, 1, 1, 1, padTop, padBottom, padLeft, padRight, 0)
            .reshape(preContractDims)
            .contract(yvec.reverse(filterReverse).shuffle(filterShuffle).reshape(filterDim), contractDims)
            .reshape(dxvec.dimensions());
}   

void DeConv2dGradXCPU(  const Tensor& x, 
                        Tensor& dx,
                        const Tensor& y,
                        const Tensor& z, 
                        const Tensor& dz,
                        bool convered,
                        int strideY,
                        int strideX) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        DeConv2dGradXCPUImpl<float>(device, 
                x.data<float>(), 
                dx.data<float>(), 
                x.shape, 
                y.data<float>(),
                y.shape,
                z.data<float>(),
                dz.data<float>(),
                z.shape,
                convered,
                strideY,
                strideX);
        break;
    case DType::Float64:
        DeConv2dGradXCPUImpl<double>(device, 
                x.data<double>(), 
                dx.data<double>(), 
                x.shape, 
                y.data<double>(),
                y.shape,
                z.data<double>(),
                dz.data<double>(),
                z.shape,
                convered,
                strideY,
                strideX);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
void DeConv2dGradYCPUImpl(  CPUDevice *device,
                            const T *x,
                            const Shape &xshape,
                            const T *y,
                            T *dy,
                            const Shape &yahspe,
                            const T *z,
                            const T *da,
                            const Shape &zshape,
                            bool convered,
                            int strideY,
                            int strideX) {
    auto eigenDevice = device->eigenDevice;

    auto batch = (int) xshape.batch;

    auto inputHeight  = (int) xshape.dim(0);
    auto inputWidth   = (int) xshape.dim(1);
    auto inputChannel = (int) xshape.dim(2);

    auto outputHeight  = (int) zshape.dim(0);
    auto outputWidth   = (int) zshape.dim(1);
    auto outputChannel = (int) zshape.dim(2);

    auto filterHeight = (int) yshape.dim(1);
    auto filterWidth  = (int) yshape.dim(2);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            dzvec(dz, batch, outputHeight, outputWidth, outputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            dyvec(dy, outputChannel, filterHeight, filterWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
            xvec(x, batch, inputHeight, inputWidth, inputChannel);

    Eigen::DSizes<int, 2> preContractDims;
    preContractDims[0] = batch * outputHeight * outputWidth;
    preContractDims[1] = filterHeight * filterWidth * inputChannel;

    Eigen::DSizes<int, 2> outputGradDim;
    outputGradDim[0] = batch * outputHeight * outputWidth;
    outputGradDim[1] = outputChannel;

    Eigen::array<Eigen::IndexPair<int>, 1> contractDims;
    contractDims[0] = Eigen::IndexPair<int>(1, 0);

    Eigen::DSizes<int, 2> shuffleDims;
    shuffleDims[0] = 1;
    shuffleDims[1] = 0;

    auto padH = std::max<int>(0, outputHeight + filterHeight - (inputHeight - 1) * strideY - 2);
    auto padW = std::max<int>(0, outputWidth + filterWidth - (inputWidth - 1) * strideX - 2);

    auto padTop = padH / 2;
    auto padBottom = padH - padTop;
    auto padLeft = padW / 2;
    auto padRight = padW - padLeft;

    dyvec.device(*device) += dzvec.reshape(outputGradDim).shuffle(shuffleDims)
                .contract(xvec.extract_image_patches(filterWidth, filterHeight, 1, 1,
                                                            1, 1, strideX,
                                                            strideY, padTop,
                                                            padBottom, padLeft, padRight,
                                                            0).reshape(preContractDims), contractDims)
                .reshape(dyvec.dimensions());
}

void DeConv2dGradYCPU(  const Tensor &x,
                        const Tensor &y, 
                        Tensor &dy,
                        const Tensor &z, 
                        const Tensor& dz,
                        bool convered,
                        int strideY,
                        int strideX) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        DeConv2dGradYCPUImpl<float>(device, 
                    x.data<float>(), 
                    x.shape, 
                    y.data<float>(), 
                    dy.data<float>(), 
                    y.shape, 
                    z.data<float>(), 
                    dz.data<float>(), 
                    z.shape,
                    convered,
                    strideY,
                    strideX);
        break;
    case DType::Float64:
        DeConv2dGradYCPUImpl<double>(device, 
                    x.data<double>(), 
                    x.shape, 
                    y.data<double>(), 
                    dy.data<double>(), 
                    y.shape, 
                    z.data<double>(), 
                    dz.data<double>(), 
                    z.shape,
                    convered,
                    strideY,
                    strideX);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

}
}