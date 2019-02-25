#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUUnaryElementWise.h"
#include "math/GPUMath.h"
#include "math/Uniform.h"
#include "math/PositiveUnitball.h"

namespace Deep8 {
namespace Math {

void positiveUnitballGPU(Tensor &x) {
    auto device = (GPUDevice*)x.device();

    Uniform(x);

    if (DType::Float32 == x.elementType.id) { 
        int size = (int)x.size();

        float sum = 0;
	    CUBLAS_CHECK(cublasSasum(device->cublasHandle, size, x.data<float>(), 1, &sum));

        float alpha = 1.0 / sum;
        CUBLAS_CHECK(cublasSscal(device->cublasHandle, size, &alpha, x.data<float>(), 1));
    } else if (DType::Float64 == x.elementType.id) {
        int size = (int)x.size();

        double sum = 0;
	    CUBLAS_CHECK(cublasDasum(device->cublasHandle, size, x.data<double>(), 1, &sum));

        double alpha = 1.0 / sum;
        CUBLAS_CHECK(cublasDscal(device->cublasHandle, size, &alpha, x.data<double>(), 1));
    } else {
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
    }
}

}
}