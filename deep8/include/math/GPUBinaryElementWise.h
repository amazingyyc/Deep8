//#ifndef DEEP8_GPUBINARYELEMENTWISE_H
//#define DEEP8_GPUBINARYELEMENTWISE_H
//
//#include "GPUBasic.h"
//
//namespace Deep8 {
//namespace Math {
//
///**
// * Binary Element Wise op
// * y cant equal or not equal z
// */
//template <typename T, typename BinaryOp>
//__global__ void BinaryElementWiseKernel(const T *x, const T *y, T *z, BinaryOp op, const int N) {
//    int start  = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//
//    for (int i = start; i < N; i += stride) {
//        z[i] = op(x[i], y[i]);
//    }
//}
//
//
//}
//}
//
//#endif
