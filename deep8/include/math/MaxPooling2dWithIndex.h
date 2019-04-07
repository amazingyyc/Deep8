#ifndef DEEP8_MATH_MAXPOOLING2DWITHINDEX_H
#define DEEP8_MATH_MAXPOOLING2DWITHINDEX_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

/**record the max index*/
void MaxPooling2dWithIndex(const Tensor &x,
                           const Tensor &index,
                          Tensor &y, 
                          bool covered = false, 
                          int filterHeight = 1, 
                          int filterWidth = 1, 
                          int strideY = 1, 
                          int strideX = 1);

void MaxPooling2dWithIndexCPU(const Tensor &x,
                              const Tensor &index,
                             Tensor &y, 
                             bool covered = false, 
                             int filterHeight = 1, 
                             int filterWidth = 1, 
                             int strideY = 1, 
                             int strideX = 1);

#ifdef HAVE_CUDA
void MaxPooling2dWithIndexGPU(const Tensor &x,
                              const Tensor &index,
                             Tensor &y, 
                             bool covered = false, 
                             int filterHeight = 1, 
                             int filterWidth = 1, 
                             int strideY = 1, 
                             int strideX = 1);
#endif


/**max pooling grad*/
void MaxPooling2dWithIndexGrad(const Tensor &x,
                      Tensor &dx,
                      const Tensor &index,
                      const Tensor &y, 
                      const Tensor &dy, 
                      bool covered = false, 
                      int filterHeight = 1, 
                      int filterWidth = 1, 
                      int strideY = 1, 
                      int strideX = 1);

void MaxPooling2dWithIndexGradCPU(const Tensor &x,
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
void MaxPooling2dWithIndexGradGPU(const Tensor &x,
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