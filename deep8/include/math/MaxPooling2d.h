#ifndef DEEP8_MATH_MAXPOOLING2D_H
#define DEEP8_MATH_MAXPOOLING2D_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void MaxPooling2d(const Tensor &x, 
                  Tensor &y, 
                  bool covered = false, 
                  int filterHeight = 1, 
                  int filterWidth = 1, 
                  int strideY = 1, 
                  int strideX = 1);

void MaxPooling2dCPU(const Tensor &x, 
                     Tensor &y, 
                     bool covered = false, 
                     int filterHeight = 1, 
                     int filterWidth = 1, 
                     int strideY = 1, 
                     int strideX = 1);

#ifdef HAVE_CUDA
void MaxPooling2dGPU(const Tensor &x, 
                     Tensor &y, 
                     bool covered = false, 
                     int filterHeight = 1, 
                     int filterWidth = 1, 
                     int strideY = 1, 
                     int strideX = 1);
#endif


/**max pooling grad*/
void MaxPooling2dGrad(const Tensor &x, 
                      Tensor &dx, 
                      const Tensor &y, 
                      const Tensor &dy, 
                      bool covered = false, 
                      int filterHeight = 1, 
                      int filterWidth = 1, 
                      int strideY = 1, 
                      int strideX = 1);

void MaxPooling2dGradCPU(const Tensor &x, 
                         Tensor &dx, 
                         const Tensor &y, 
                         const Tensor &dy, 
                         bool covered = false, 
                         int filterHeight = 1, 
                         int filterWidth = 1, 
                         int strideY = 1, 
                         int strideX = 1);

#ifdef HAVE_CUDA
void MaxPooling2dGradGPU(const Tensor &x, 
                         Tensor &dx, 
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