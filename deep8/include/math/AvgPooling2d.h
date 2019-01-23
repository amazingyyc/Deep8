#ifndef DEEP8_MATH_AVGPOOLING2D_H
#define DEEP8_MATH_AVGPOOLING2D_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**avg pooling 2d*/
void AvgPooling2d(const Tensor &x, 
                  Tensor &y, 
                  bool coverd = false, 
                  int filterHeight = 1, 
                  int filterWidth = 1, 
                  int strideY = 1, 
                  int strideX = 1);

void AvgPooling2dCPU(const Tensor &x, 
                     Tensor &y, 
                     bool coverd = false, 
                     int filterHeight = 1, 
                     int filterWidth = 1, 
                     int strideY = 1, 
                     int strideX = 1);

#ifdef HAVE_CUDA
void AvgPooling2dGPU(const Tensor &x, 
                     Tensor &y, 
                     bool coverd = false, 
                     int filterHeight = 1, 
                     int filterWidth = 1, 
                     int strideY = 1, 
                     int strideX = 1);
#endif


/**avg pooling grad*/
void AvgPooling2dGrad(const Tensor &x, 
                      Tensor &dx, 
                      const Tensor &y, 
                      const Tensor &dy, 
                      bool coverd = false, 
                      int filterHeight = 1, 
                      int filterWidth = 1, 
                      int strideY = 1, 
                      int strideX = 1);

void AvgPooling2dGradCPU(const Tensor &x, 
                         Tensor &dx, 
                         const Tensor &y, 
                         const Tensor &dy, 
                         bool coverd = false, 
                         int filterHeight = 1, 
                         int filterWidth = 1, 
                         int strideY = 1, 
                         int strideX = 1);

#ifdef HAVE_CUDA
void AvgPooling2dGradGPU(const Tensor &x, 
                         Tensor &dx, 
                         const Tensor &y, 
                         const Tensor &dy,
                         bool coverd = false, 
                         int filterHeight = 1, 
                         int filterWidth = 1, 
                         int strideY = 1, 
                         int strideX = 1);
#endif

}
}

#endif