#include "math/Constant.h"

namespace Deep8 {
namespace Math {

void Constant(Tensor &x, float scalar) {
    if (DeviceType::CPU == x.deviceType()) {
        ConstantCPU(x, scalar);
    } else {
#ifdef HAVE_CUDA
        ConstantGPU(x, scalar);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void ConstantCPU(Tensor &x, float scalar) {
    auto eigenDevice = ((CPUDevice*)x.device())->eigenDevice;

    if (DType::Float32 == x.elementType.id) {
        Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> xvec(x.data<float>(), (int)x.size());
        xvec.setConstant(scalar);
    } else if (DType::Float64 == x.elementType.id) {
        Eigen::TensorMap<Eigen::Tensor<double, 1, Eigen::RowMajor>> xvec(x.data<double>(), (int)x.size());
        xvec.setConstant(double(scalar));
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
    }
}




}
}