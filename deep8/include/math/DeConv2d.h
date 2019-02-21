#ifndef DEEP8_MATH_DECONV2D_H
#define DEEP8_MATH_DECONV2D_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void DeConv2d(  const Tensor &x, 
                const Tensor &y, 
                Tensor &z,
                void *zmat,
                bool convered = false,
                int strideY = 1,
                int strideX = 1);

void DeConv2dCPU(   const Tensor &x, 
                    const Tensor &y, 
                    Tensor &z,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1);

#ifdef HAVE_CUDA
void DeConv2dGPU(   const Tensor &x, 
                    const Tensor &y, 
                    Tensor &z,
                    void *zmat,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1);
#endif

void DeConv2dGradX( const Tensor& x, 
                    Tensor& dx,
                    const Tensor& y,
                    const Tensor& z, 
                    const Tensor& dz,
                    void *dzmat,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1);

void DeConv2dGradXCPU(  const Tensor& x, 
                        Tensor& dx,
                        const Tensor& y,
                        const Tensor& z, 
                        const Tensor& dz,
                        bool convered = false,
                        int strideY = 1,
                        int strideX = 1);

#ifdef HAVE_CUDA
void DeConv2dGradXGPU(  const Tensor& x, 
                        Tensor& dx,
                        const Tensor& y,
                        const Tensor& z, 
                        const Tensor& dz,
                        void *dzmat,
                        bool convered = false,
                        int strideY = 1,
                        int strideX = 1);
#endif

/**gradient for y*/
void DeConv2dGradY( const Tensor &x,
                    const Tensor &y, 
                    Tensor &dy,
                    const Tensor &z, 
                    const Tensor& dz,
                    void *dzmat,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1);

void DeConv2dGradYCPU(  const Tensor &x,
                        const Tensor &y, 
                        Tensor &dy,
                        const Tensor &z, 
                        const Tensor& dz,
                        bool convered = false,
                        int strideY = 1,
                        int strideX = 1);

#ifdef HAVE_CUDA
void DeConv2dGradYGPU(  const Tensor &x,
                        const Tensor &y, 
                        Tensor &dy,
                        const Tensor &z, 
                        const Tensor& dz,
                        void *dzmat,
                        bool convered = false,
                        int strideY = 1,
                        int strideX = 1);
#endif


}
}

#endif