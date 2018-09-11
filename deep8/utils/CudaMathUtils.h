#ifndef DEEP8_CUDAMATHUTILS_H
#define DEEP8_CUDAMATHUTILS_H

#include "Macro.h"

#ifdef HAVE_CUDA

namespace Deep8 {

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuExp(const real &in) {
	using ::exp;
	return exp(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuExp(const float &in) {
	return expf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuExp(const double &in) {
	return exp(in);
}

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuAbs(const real &in) {
	using ::abs;
	return abs(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuAbs(const float &in) {
	return fabsf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuAbs(const double &in) {
	return fabs(in);
}

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuSqrt(const real &in) {
	using ::sqrt;
	return sqrt(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuSqrt(const float &in) {
	return sqrtf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuSqrt(const double &in) {
	return sqrt(in);
}

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuLog(const real &in) {
	using ::log;
	return log(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuLog(const float &in) {
	return logf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuLog(const double &in) {
	return log(in);
}


template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuPow(const real &in, const real &scalar) {
	using ::pow;
	return pow(in, scalar);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuPow(const float &in, const float &scalar) {
	return powf(in, scalar);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuPow(const double &in, const double &scalar) {
	return pow(in, scalar);
}

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuTanh(const real &in) {
	using ::tanh;
	return tanh(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuTanh(const float &in) {
	return tanhf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuTanh(const double &in) {
	return tanh(in);
}

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuMax(const real &i1, const real &i2) {
	using ::max;
	return max(i1, i2);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuMax(const float &i1, const float &i2) {
	return fmaxf(i1, i2);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuMax(const double &i1, const double &i2) {
	return fmax(i1, i2);
}

/**
 * the CUDN does not support template shared memory
 * ref:https://wrf.ecse.rpi.edu//wiki/ParallelComputingSpring2015/cuda/nvidia/samples/0_Simple/simpleTemplates/sharedmem.cuh
 */
template <typename T>
struct SharedMemory {
	__device__ T *pointer() {
		return nullptr;
	}
};

template <>
struct SharedMemory<float> {
	__device__ float *pointer() {
		extern __shared__ float sharedFloat[];
		return sharedFloat;
	}
};

template <>
struct SharedMemory<double> {
	__device__ double *pointer() {
		extern __shared__ double sharedDouble[];
		return sharedDouble;
	}
};

template <unsigned int blockSize, typename real>
__device__ DEEP8_CUDA_INLINE void warpSumReduce(volatile real *shared, int threaId) {
	if (blockSize >= 64) shared[threaId] += shared[threaId + 32];
	if (blockSize >= 32) shared[threaId] += shared[threaId + 16];
	if (blockSize >= 16) shared[threaId] += shared[threaId +  8];
	if (blockSize >=  8) shared[threaId] += shared[threaId +  4];
	if (blockSize >=  4) shared[threaId] += shared[threaId +  2];
	if (blockSize >=  2) shared[threaId] += shared[threaId +  1];
}

template <unsigned int blockSize, typename real>
__device__ DEEP8_CUDA_INLINE void warpMaxReduce(volatile real *shared, int threaId) {
	if (blockSize >= 64) { shared[threaId] = cuMax(shared[threaId], shared[threaId + 32]); }
	if (blockSize >= 32) { shared[threaId] = cuMax(shared[threaId], shared[threaId + 16]); }
	if (blockSize >= 16) { shared[threaId] = cuMax(shared[threaId], shared[threaId +  8]); }
	if (blockSize >= 8)  { shared[threaId] = cuMax(shared[threaId], shared[threaId +  4]); }
	if (blockSize >= 4)  { shared[threaId] = cuMax(shared[threaId], shared[threaId +  2]); }
	if (blockSize >= 2)  { shared[threaId] = cuMax(shared[threaId], shared[threaId +  1]); }
}

}

#endif

#endif