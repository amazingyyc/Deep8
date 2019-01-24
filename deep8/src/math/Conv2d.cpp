#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "utils/GPUMathUtils.h"
#include "math/Conv2d.h"

namespace Deep8 {
namespace Math {

template <typename T>
void Conv2dCPUImpl(CPUDevice *device,
                   const T* x, const Shape& xshape,
                   const T* y, const Shape& yshape,
                   T* z, const Shape& zshape,
                   bool convered,
                   int strideY,
                   int strideX,
                   int dilationY,
                   int dilationX) {
    auto eigenDevice = device->eigenDevice;

    auto batch = (int)x.batch();

    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto filterHeight = (int)y.dim(1);
    auto filterWidth  = (int)y.dim(2);

    auto outputHeight  = (int)z.dim(0);
    auto outputWidth   = (int)z.dim(1);
    auto outputChannel = (int)z.dim(2);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        xTensor(x, batch, inputHeight, inputWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        yTensor(y, outputChannel, filterHeight, filterWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        zTensor(z, batch, outputHeight, outputWidth, outputChannel);

    Eigen::DSizes<int, 2> preContractDims;
    preContractDims[0] = batch * outputHeight * outputWidth;
    preContractDims[1] = filterHeight * filterWidth * inputChannel;

    Eigen::DSizes<int, 2> shuffleDims;
    shuffleDims[0] = 1;
    shuffleDims[1] = 0;

    Eigen::array<Eigen::IndexPair<int>, 1> contractDims;
    contractDims[0] = Eigen::IndexPair<int>(1, 0);

    Eigen::DSizes<int, 2> kernelDims;
    kernelDims[0] = outputChannel;
    kernelDims[1] = filterHeight * filterWidth * inputChannel;

    auto realFilterHeight = filterHeight + (filterHeight - 1) * (dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * (dilationX - 1);

    auto padY = std::max<int>(0, (outputHeight - 1) * strideY + realFilterHeight - inputHeight);
    auto padX = std::max<int>(0, (outputWidth  - 1) * strideX + realFilterWidth  - inputWidth);

    auto padTop    = padY / 2;
    auto padBottom = padY - padTop;
    auto padLeft   = padX / 2;
    auto padRight  = padX - padLeft;

    zTensor.device(*eigenDevice) = xTensor.extract_image_patches(
        filterWidth, filterHeight, strideX, strideY, dilationX, dilationY, 1, 1, padTop, padBottom, padLeft, padRight, 0)
        .reshape(preContractDims)
        .contract(yTensor.reshape(kernelDims).shuffle(shuffleDims), contractDims)
        .reshape(zTensor.dimensions());
}

void Conv2dCPU(const Tensor &x, const Tensor &y, Tensor &z, 
               bool convered, 
               int strideY, 
               int strideX, 
               int dilationY, 
               int dilationX) {
    if (DType::Float32 == x.type.id) {
        Conv2dCPUImpl<float>(x.device(), 
                             x.data<float>(), x.shape, 
                             y.data<float>(), y.shape, 
                             z.data<float>(), z.shape, 
                             convered, 
                             strideY, 
                             strideX, 
                             dilationY, 
                             dilationX);
    } else if (DType::Float64 == x.type.id) {
        Conv2dCPUImpl<double>(x.device(),
                             x.data<double>(), x.shape,
                             y.data<double>(), y.shape,
                             z.data<double>(), z.shape,
                             convered,
                             strideY,
                             strideX,
                             dilationY,
                             dilationX);
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
    }
}

/**calculate the gradient for x (input)*/

template <typename T>
void Conv2dGradXCPUImpl(CPUDevice *device,
                        const T* x, T* dx,
                        const T* y,
                        const T* z, const T* dz,
                        int batch,
                        int inputHeight,
                        int inputWidth,
                        int inputChannel,
                        int outputHeight,
                        int outputWidth,
                        int outputChannel,
                        int filterHeight,
                        int filterWidth,
                        bool convered,
                        int strideY,
                        int strideX,
                        int dilationY,
                        int dilationX) {
    auto eigenDeviec = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        dzTensor(dz, batch, outputHeight, outputWidth, outputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        dxTensor(dx, batch, inputHeight, inputWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        yTensor(y, outputChannel, filterHeight, filterWidth, inputChannel);

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

    auto realFilterHeight = filterHeight + (filterHeight - 1) * (dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * (dilationX - 1);

    auto forwardPadTop  = std::max<int>(0, ((outputHeight - 1) * strideY + realFilterHeight - inputHeight) / 2);
    auto forwardPadLeft = std::max<int>(0, ((outputWidth - 1)  * strideX + realFilterWidth  - inputWidth)  / 2);

    auto padTop  = realFilterHeight - 1 - forwardPadTop;
    auto padLeft = realFilterWidth  - 1 - forwardPadLeft;
    auto padBottom = inputHeight - (outputHeight - 1) * strideY - 2 - padTop  + realFilterHeight;
    auto padRight  = inputWidth  - (outputWidth  - 1) * strideX - 2 - padLeft + realFilterWidth;

    dxTensor.device(*eigenDevice) += dzTensor.extract_image_patches(
        filterWidth, filterHeight, 1, 1, dilationX, dilationY, strideX, strideY, padTop, padBottom, padLeft, padRight, 0)
        .reshape(preContractDims)
        .contract(yTensor.reverse(filterReverse).shuffle(filterShuffle).reshape(filterDim), contractDims)
        .reshape(dxTensor.dimensions());
}

void Conv2dGradXCPU(const Tensor& x, Tensor& dx,
                    const Tensor& y,
                    const Tensor& z, const Tensor& dz,
                    bool convered,
                    int strideY,
                    int strideX,
                    int dilationY,
                    int dilationX) {
    auto batch = (int)x.batch();

    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto filterHeight = (int)y.dim(1);
    auto filterWidth  = (int)y.dim(2);

    auto outputHeight  = (int)z.dim(0);
    auto outputWidth   = (int)z.dim(1);
    auto outputChannel = (int)z.dim(2);

    if (DType::Float32 == x.type.id) {
        Conv2dGradXCPUImpl<float>(x.device(),
                                  x.data<float>(),
                                  dx.data<float>(),
                                  y.data<float>(),
                                  z.data<float>(),
                                  dz.data<float>(),
                                  batch,
                                  inputHeight,
                                  inputWidth,
                                  inputChannel,
                                  outputHeight,
                                  outputWidth,
                                  outputChannel,
                                  filterHeight,
                                  filterWidth,
                                  convered,
                                  strideY,
                                  strideX,
                                  dilationY,
                                  dilationX);
    } else if (DType::Float64 == x.type.id) {
        Conv2dGradXCPUImpl<double>(x.device(),
                                  x.data<double>(),
                                  dx.data<double>(),
                                  y.data<double>(),
                                  z.data<double>(),
                                  dz.data<double>(),
                                  batch,
                                  inputHeight,
                                  inputWidth,
                                  inputChannel,
                                  outputHeight,
                                  outputWidth,
                                  outputChannel,
                                  filterHeight,
                                  filterWidth,
                                  convered,
                                  strideY,
                                  strideX,
                                  dilationY,
                                  dilationX);
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
    }
}

/**grad for Y*/
template <typename T>
void Conv2dGradYCPUImpl(CPUDevice* device,
                        const T* x,
                        const T* y, T* dy,
                        const T* z, const T* dz,
                        int batch,
                        int inputHeight,
                        int inputWidth,
                        int inputChannel,
                        int outputHeight,
                        int outputWidth,
                        int outputChannel,
                        int filterHeight,
                        int filterWidth,
                        bool convered,
                        int strideY,
                        int strideX,
                        int dilationY,
                        int dilationX) {
    auto eigenDeviec = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        dzTensor(dz, batch, outputHeight, outputWidth, outputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        dyTensor(dy, outputChannel, filterHeight, filterWidth, inputChannel);

    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
        xTensor(x, batch, inputHeight, inputWidth, inputChannel);

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

    int realFilterH = filterHeight + (filterHeight - 1) * (dilationY - 1);
    int realFilterW = filterWidth  + (filterWidth  - 1) * (dilationX - 1);

    int padY = std::max<int>(0, (outputHeight - 1) * strideY + realFilterH - inputHeight);
    int padX = std::max<int>(0, (outputWidth  - 1) * strideX + realFilterW - inputWidth);

    int padTop    = padY / 2;
    int padBottom = padY - padTop;
    int padLeft   = padX / 2;
    int padRight  = padX - padLeft;

    dyTensor.device(*eigenDevice) += dzTensor.reshape(outputGradDim).shuffle(shuffleDims)
        .contract(xTensor.extract_image_patches(filterWidth, filterHeight, strideX,
                                                    strideY, dilationX, dilationY, 1, 1,
                                                    padTop, padBottom, padLeft, padRight,
                                                    0).reshape(preContractDims), contractDims)
        .reshape(dyTensor.dimensions());
}

void Conv2dGradYCPU(const Tensor &x,
                    const Tensor &y, Tensor &dy,
                    const Tensor &z, const Tensor& dz,
                    bool convered,
                    int strideY,
                    int strideX,
                    int dilationY,
                    int dilationX) {
    auto batch = (int)x.batch();

    auto inputHeight  = (int)x.dim(0);
    auto inputWidth   = (int)x.dim(1);
    auto inputChannel = (int)x.dim(2);

    auto filterHeight = (int)y.dim(1);
    auto filterWidth  = (int)y.dim(2);

    auto outputHeight  = (int)z.dim(0);
    auto outputWidth   = (int)z.dim(1);
    auto outputChannel = (int)z.dim(2);

    if (DType::Float32 == x.type.id) {
        Conv2dGradYCPUImpl<float>(x.device(),
                                  x.data<float>(),
                                  y.data<float>(),
                                  dy.data<float>(),
                                  z.data<float>(),
                                  dz.data<float>(),
                                  batch,
                                  inputHeight,
                                  inputWidth,
                                  inputChannel,
                                  outputHeight,
                                  outputWidth,
                                  outputChannel,
                                  filterHeight,
                                  filterWidth,
                                  convered,
                                  strideY,
                                  strideX,
                                  dilationY,
                                  dilationX);
    } else if (DType::Float64 == x.type.id) {
        Conv2dGradYCPUImpl<double>(x.device(),
                                  x.data<double>(),
                                  y.data<double>(),
                                  dy.data<double>(),
                                  z.data<double>(),
                                  dz.data<double>(),
                                  batch,
                                  inputHeight,
                                  inputWidth,
                                  inputChannel,
                                  outputHeight,
                                  outputWidth,
                                  outputChannel,
                                  filterHeight,
                                  filterWidth,
                                  convered,
                                  strideY,
                                  strideX,
                                  dilationY,
                                  dilationX);
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
    }
}

}
}