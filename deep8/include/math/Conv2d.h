#ifndef DEEP8_MATH_CONV2D_H
#define DEEP8_MATH_CONV2D_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**
 * conv2d
 * the ptr is the temp memory, when run on CPU it should be nullptr
 * in GPU it should point to a memory and size is
 * (sizeof(dataType) * batch * outputHeight * outputWidth * filterHeight * filterWidth * inputChannel)
 */
size_t Conv2dInterimSize(const Tensor& x, const Tensor& y, Tensor& z, bool covered, int strideY, int strideX, int dilationY, int dilationX);

void Conv2d(const Tensor &x, const Tensor &y, Tensor &z, bool covered, int strideY, int strideX, int dilationY, int dilationX, void* interimPtr);

void Conv2dCPU(const Tensor& x, const Tensor& y, Tensor& z, bool covered, int strideY, int strideX, int dilationY, int dilationX);

#ifdef HAVE_CUDA
void Conv2dGPU(const Tensor& x, const Tensor& y, Tensor& z, bool covered, int strideY, int strideX, int dilationY, int dilationX, void* xcol);
#endif

/**gradient for x*/
size_t Conv2dGradXInterimSize(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX);

void Conv2dGradX(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX, void* interimPtr);

void Conv2dGradXCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX);

#ifdef HAVE_CUDA
void Conv2dGradXGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX, void* dxcol);
#endif

/**gradient for y*/
size_t Conv2dGradYInterimSize(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX);

void Conv2dGradY(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX, void* interimPtr);

void Conv2dGradYCPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX);

#ifdef HAVE_CUDA
void Conv2dGradYGPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, int dilationY, int dilationX,void* xcol);
#endif

}
}

#endif