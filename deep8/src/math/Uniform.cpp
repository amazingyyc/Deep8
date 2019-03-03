#include "math/Uniform.h"

namespace Deep8 {
namespace Math {
    
void Uniform(Tensor &x, float left, float right) {
    if (DeviceType::CPU == x.deviceType()) {
        UniformCPU(x, left, right);
    } else {
#ifdef HAVE_CUDA
        UniformGPU(x, left, right);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void UniformCPU(Tensor &x, float left, float right) {
	auto device = ((CPUDevice*)x.device());

	if (DType::Float32 == x.elementType.id) {
		std::uniform_real_distribution<float> distribution(left, right);

		std::generate(x.data<float>(), x.data<float>() + x.size(), std::bind(distribution, device->randGenerator));
	} else if (DType::Float64 == x.elementType.id) {
		std::uniform_real_distribution<double> distribution(left, right);

		std::generate(x.data<double>(), x.data<double>() + x.size(), std::bind(distribution, device->randGenerator));
	} else {
		DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
	}
}

}
}