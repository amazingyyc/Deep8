#include "math/Uniform.h"
#include "math/PositiveUnitball.h"

namespace Deep8 {
namespace Math {

void positiveUnitball(Tensor &x) {
    if (DeviceType::CPU == x.deviceType()) {
        positiveUnitballCPU(x);
    } else {
#ifdef HAVE_CUDA
        positiveUnitballGPU(x);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void positiveUnitballCPU(Tensor &x) {
    Uniform(x, 0, 1);

    auto eigenDevice = ((CPUDevice*)x.device())->eigenDevice;

    if (DType::Float32 == x.elementType.id) { 
        float sum = 0;

		Eigen::array<int, 1> sumDims = { 0 };

		Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> sumvec(&sum, 1);
		Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> xvec(x.data<float>(), (int)x.size());
		
		sumvec.device(*eigenDevice) = xvec.sum(sumDims);

		xvec.device(*eigenDevice) = xvec / sum;
    } else if (DType::Float64 == x.elementType.id) {
        double sum = 0;

		Eigen::array<int, 1> sumDims = { 0 };

		Eigen::TensorMap<Eigen::Tensor<double, 1, Eigen::RowMajor>> sumvec(&sum, 1);
		Eigen::TensorMap<Eigen::Tensor<double, 1, Eigen::RowMajor>> xvec(x.data<double>(), (int)x.size());
		
		sumvec.device(*eigenDevice) = xvec.sum(sumDims);

		xvec.device(*eigenDevice) = xvec / sum;
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
    }
}

}
}