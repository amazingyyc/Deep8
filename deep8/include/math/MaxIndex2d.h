#ifndef DEEP8_MATH_MAXINDEX2D_H
#define DEEP8_MATH_MAXINDEX2D_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Tensor.h"

namespace Deep8 {
namespace Math {

void MaxIndex2d(const Tensor &x,
                Tensor &index,
                bool covered = true, 
                int filterHeight = 1, 
                int filterWidth = 1, 
                int strideY = 1, 
                int strideX = 1);

void MaxIndex2dCPU(const Tensor &x,
                    Tensor &index,
                    bool covered = true, 
                    int filterHeight = 1, 
                    int filterWidth = 1, 
                    int strideY = 1, 
                    int strideX = 1);

#ifdef HAVE_CUDA
void MaxIndex2dGPU(const Tensor &x,
                    Tensor &index,
                    bool covered = true, 
                    int filterHeight = 1, 
                    int filterWidth = 1, 
                    int strideY = 1, 
                    int strideX = 1);
#endif

}
}

#endif