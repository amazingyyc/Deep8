#ifndef DEEP8_GPUMATHUTILS_H
#define DEEP8_GPUMATHUTILS_H

#include "../basic/GPUBasic.h"

#ifdef HAVE_CUDA

namespace Deep8 {

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuExp(const real &in) {
	//using ::exp;
	//return exp(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuExp(const float &in) {
	return expf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuExp(const double &in) {
	return exp(in);
}

#ifdef HAVE_HALF

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuExp(const half &in) {
	return hexp(in);
}

#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuAbs(const real &in) {
	//using ::abs;
	//return abs(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuAbs(const float &in) {
	return fabsf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuAbs(const double &in) {
	return fabs(in);
}

#ifdef HAVE_HALF

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuAbs(const half &in) {
	return in >= half(0) ? in : -in;
}

#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuSqrt(const real &in) {
	/*using ::sqrt;
	return sqrt(in);*/
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuSqrt(const float &in) {
	return sqrtf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuSqrt(const double &in) {
	return sqrt(in);
}

#ifdef HAVE_HALF

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuSqrt(const half &in) {
	return hsqrt(in);
}

#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuLog(const real &in) {
	//using ::log;
	//return log(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuLog(const float &in) {
	return logf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuLog(const double &in) {
	return log(in);
}

#ifdef HAVE_HALF

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuLog(const half &in) {
	return hlog(in);
}

#endif


template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuPow(const real &in, const real &scalar) {
	/*using ::pow;
	return pow(in, scalar);*/
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuPow(const float &in, const float &scalar) {
	return powf(in, scalar);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuPow(const double &in, const double &scalar) {
	return pow(in, scalar);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuPow(const half &in, const half &scalar) {
	return __float2half(powf(__half2float(in), __half2float(scalar)));
}
#endif


template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuTanh(const real &in) {
	/*using ::tanh;
	return tanh(in);*/
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuTanh(const float &in) {
	return tanhf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuTanh(const double &in) {
	return tanh(in);
}

#ifdef HAVE_HALF

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuTanh(const half &in) {
	return __float2half(tanh(__half2float(in)));
}

#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuMax(const real &i1, const real &i2) {
	/*using ::max;
	return max(i1, i2);*/
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuMax(const float &i1, const float &i2) {
	return fmaxf(i1, i2);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuMax(const double &i1, const double &i2) {
	return fmax(i1, i2);
}

#ifdef HAVE_HALF

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuMax(const half &i1, const half &i2) {
	return i1 >= i2 ? i1 : i2;
}

#endif

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

#ifdef HAVE_HALF

template <>
struct SharedMemory<half> {
	__device__ half *pointer() {
		extern __shared__ half sharedHalf[];
		return sharedHalf;
	}
};

#endif

template <unsigned int blockSize, typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE void warpSumReduce(volatile real *shared, int threadId) {
	if (blockSize >= 64) shared[threadId] += shared[threadId + 32];
	if (blockSize >= 32) shared[threadId] += shared[threadId + 16];
	if (blockSize >= 16) shared[threadId] += shared[threadId +  8];
	if (blockSize >=  8) shared[threadId] += shared[threadId +  4];
	if (blockSize >=  4) shared[threadId] += shared[threadId +  2];
	if (blockSize >=  2) shared[threadId] += shared[threadId +  1];
}


#ifdef HAVE_HALF

template <unsigned int blockSize, typename real = half>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE void warpSumReduce(volatile half *shared, int threadId) {
	if (blockSize >= 64) shared[threadId] = __hadd(shared[threadId], shared[threadId + 32]);
	if (blockSize >= 32) shared[threadId] = __hadd(shared[threadId], shared[threadId + 16]);
	if (blockSize >= 16) shared[threadId] = __hadd(shared[threadId], shared[threadId +  8]);
	if (blockSize >=  8) shared[threadId] = __hadd(shared[threadId], shared[threadId +  4]);
	if (blockSize >=  4) shared[threadId] = __hadd(shared[threadId], shared[threadId +  2]);
	if (blockSize >=  2) shared[threadId] = __hadd(shared[threadId], shared[threadId +  1]);
}

#endif

template <unsigned int blockSize, typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE void warpMaxReduce(volatile real *shared, int threadId) {
	if (blockSize >= 64) { shared[threadId] = cuMax(shared[threadId], shared[threadId + 32]); }
	if (blockSize >= 32) { shared[threadId] = cuMax(shared[threadId], shared[threadId + 16]); }
	if (blockSize >= 16) { shared[threadId] = cuMax(shared[threadId], shared[threadId +  8]); }
	if (blockSize >= 8)  { shared[threadId] = cuMax(shared[threadId], shared[threadId +  4]); }
	if (blockSize >= 4)  { shared[threadId] = cuMax(shared[threadId], shared[threadId +  2]); }
	if (blockSize >= 2)  { shared[threadId] = cuMax(shared[threadId], shared[threadId +  1]); }
}

}

#endif

#endif