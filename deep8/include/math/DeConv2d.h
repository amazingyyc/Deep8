#ifndef DEEP8_MATH_DECONV2D_H
#define DEEP8_MATH_DECONV2D_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

size_t DeConv2dInterimSize(const Tensor& x, const Tensor& y, Tensor& z, bool covered, int strideY, int strideX);

void DeConv2d(const Tensor& x, const Tensor& y, Tensor& z, bool covered, int strideY, int strideX, void* interimPtr);

void DeConv2dCPU(const Tensor& x, const Tensor& y, Tensor& z, bool covered, int strideY, int strideX);

#ifdef HAVE_CUDA
void DeConv2dGPU(const Tensor& x, const Tensor& y, Tensor& z, bool covered, int strideY, int strideX, void* interimPtr);
#endif

size_t DeConv2dGradXInterimSize(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX);

void DeConv2dGradX(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, void* interimPtr);

void DeConv2dGradXCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX);

#ifdef HAVE_CUDA
void DeConv2dGradXGPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, void* interimPtr);
#endif

size_t DeConv2dGradYInterimSize(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX);

/**gradient for y*/
void DeConv2dGradY(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, void* interimPtr);

void DeConv2dGradYCPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX);

#ifdef HAVE_CUDA
void DeConv2dGradYGPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz, bool covered, int strideY, int strideX, void* interimPtr);
#endif

}
}

#endif