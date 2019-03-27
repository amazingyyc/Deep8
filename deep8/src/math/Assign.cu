#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUUnaryElementWise.h"
#include "math/GPUMath.h"
#include "math/Assign.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct AssignKernelOp {
    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x) {
        return x;
    }
};

void AssignGPU(const Tensor &x, Tensor &y) {
    auto n = (int) x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (n + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        UnaryElementWiseKernel<float, AssignKernelOp<float>> << <grideSize, blockSize >> > (x, y, AssignKernelOp<float>(), n);
        break;
    case DType::Float64:
        UnaryElementWiseKernel<double, AssignKernelOp<double>> << <grideSize, blockSize >> > (x, y, AssignKernelOp<double>(), n);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        UnaryElementWiseKernel<half, AssignKernelOp<half>> << <grideSize, blockSize >> > (x, y, AssignKernelOp<half>(), n);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}