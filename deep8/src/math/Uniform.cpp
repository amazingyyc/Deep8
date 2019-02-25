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

void UniformCPU(Tensor &tensor, float left, float right) {
	auto device = ((CPUDevice*)tensor.device());

	if (DType::Float32 == tensor.elementType.id) {
		std::uniform_real_distribution<float> distribution(left, right);

		std::generate(tensor.data<float>(), tensor.data<float>() + tensor.size(), std::bind(distribution, device->randGenerator));
	} else if (DType::Float64 == tensor.elementType.id) {
		std::uniform_real_distribution<double> distribution(left, right);

		std::generate(tensor.data<double>(), tensor.data<double>() + tensor.size(), std::bind(distribution, device->randGenerator));
	} else {
		DEEP8_RUNTIME_ERROR("type " << tensor.elementType.name << " is not support");
	}
}

}
}