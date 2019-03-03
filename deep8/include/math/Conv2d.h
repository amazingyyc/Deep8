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
 *  (sizeof(dataType) * batch * outputHeight * outputWidth * filterHeight * filterWidth * inputChannel)
 */
void Conv2d(const Tensor &x, 
            const Tensor &y, 
            Tensor &z,
            void *ptr,
            bool convered = false,
            int strideY = 1,
            int strideX = 1,
            int dilationY = 1,
            int dilationX = 1);

void Conv2dCPU( const Tensor &x, 
                const Tensor &y, 
                Tensor &z,
                bool convered = false,
                int strideY = 1,
                int strideX = 1,
                int dilationY = 1,
                int dilationX = 1);

#ifdef HAVE_CUDA
void Conv2dGPU( const Tensor &x, 
                const Tensor &y, 
                Tensor &z,
                void *xcol,
                bool convered = false,
                int strideY = 1,
                int strideX = 1,
                int dilationY = 1,
                int dilationX = 1);
#endif

/**gradient for x*/
void Conv2dGradX(const Tensor& x, 
                Tensor& dx,
                const Tensor& y,
                const Tensor& z, 
                const Tensor& dz,
                void *ptr,
                bool convered = false,
                int strideY = 1,
                int strideX = 1,
                int dilationY = 1,
                int dilationX = 1);

void Conv2dGradXCPU(const Tensor& x, 
                    Tensor& dx,
                    const Tensor& y,
                    const Tensor& z, 
                    const Tensor& dz,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1,
                    int dilationY = 1,
                    int dilationX = 1);

#ifdef HAVE_CUDA
void Conv2dGradXGPU(const Tensor& x, 
                    Tensor& dx,
                    const Tensor& y,
                    const Tensor& z, 
                    const Tensor& dz,
                    void *dxcol,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1,
                    int dilationY = 1,
                    int dilationX = 1);
#endif

/**gradient for y*/
void Conv2dGradY(const Tensor &x,
                const Tensor &y, 
                Tensor &dy,
                const Tensor &z, 
                const Tensor& dz,
                void *ptr,
                bool convered = false,
                int strideY = 1,
                int strideX = 1,
                int dilationY = 1,
                int dilationX = 1);

void Conv2dGradYCPU(const Tensor &x,
                    const Tensor &y, 
                    Tensor &dy,
                    const Tensor &z, 
                    const Tensor& dz,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1,
                    int dilationY = 1,
                    int dilationX = 1);

#ifdef HAVE_CUDA
void Conv2dGradYGPU(const Tensor &x,
                    const Tensor &y, 
                    Tensor &dy,
                    const Tensor &z, 
                    const Tensor& dz,
                    void *xcol,
                    bool convered = false,
                    int strideY = 1,
                    int strideX = 1,
                    int dilationY = 1,
                    int dilationX = 1);
#endif

}
}

#endif