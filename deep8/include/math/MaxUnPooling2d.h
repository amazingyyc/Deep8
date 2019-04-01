#ifndef DEEP8_MATH_MAXUNPOOLING2D_H
#define DEEP8_MATH_MAXUNPOOLING2D_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void MaxUnPooling2d(const Tensor &x, 
                  const Tensor &index,
                  Tensor &y, 
                  bool covered = false, 
                  int filterHeight = 1, 
                  int filterWidth = 1, 
                  int strideY = 1, 
                  int strideX = 1);

void MaxUnPooling2dCPU(const Tensor &x,
                     const Tensor &index,
                     Tensor &y, 
                     bool covered = false, 
                     int filterHeight = 1, 
                     int filterWidth = 1, 
                     int strideY = 1, 
                     int strideX = 1);

#ifdef HAVE_CUDA
void MaxUnPooling2dGPU(const Tensor &x,
                     const Tensor &index,
                     Tensor &y, 
                     bool covered = false, 
                     int filterHeight = 1, 
                     int filterWidth = 1, 
                     int strideY = 1, 
                     int strideX = 1);
#endif


/**max pooling grad*/
void MaxUnPooling2dGrad(const Tensor &x,
                      Tensor &dx,
                      const Tensor &index,
                      const Tensor &y, 
                      const Tensor &dy, 
                      bool covered = false, 
                      int filterHeight = 1, 
                      int filterWidth = 1, 
                      int strideY = 1, 
                      int strideX = 1);

void MaxUnPooling2dGradCPU(const Tensor &x,
                         Tensor &dx, 
                         const Tensor &index,
                         const Tensor &y, 
                         const Tensor &dy, 
                         bool covered = false, 
                         int filterHeight = 1, 
                         int filterWidth = 1, 
                         int strideY = 1, 
                         int strideX = 1);

#ifdef HAVE_CUDA
void MaxUnPooling2dGradGPU(const Tensor &x,
                         Tensor &dx, 
                         const Tensor &index,
                         const Tensor &y, 
                         const Tensor &dy,
                         bool covered = false, 
                         int filterHeight = 1, 
                         int filterWidth = 1, 
                         int strideY = 1, 
                         int strideX = 1);
#endif

}
}

#endif