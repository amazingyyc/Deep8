#include "math/Gaussian.h"

namespace Deep8 {
namespace Math {

void Gaussian(Tensor &x, float mean, float stddev) {
    if (DeviceType::CPU == x.deviceType()) {
        GaussianCPU(x, mean, stddev);
    } else {
#ifdef HAVE_CUDA
        GaussianGPU(x, mean, stddev);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void GaussianCPU(Tensor &x, float mean, float stddev) {
    auto device = ((CPUDevice*)x.device());

	if (DType::Float32 == x.elementType.id) {
		std::normal_distribution<float> distribution(mean, stddev);

		std::generate(x.data<float>(), x.data<float>() + x.size(), std::bind(distribution, device->randGenerator));
	} else if (DType::Float64 == x.elementType.id) {
		std::normal_distribution<double> distribution(mean, stddev);

		std::generate(x.data<double>(), x.data<double>() + x.size(), std::bind(distribution, device->randGenerator));
	} else {
		DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
	}
}


}
}