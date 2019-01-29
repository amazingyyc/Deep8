#ifndef DEEP8_GPUMATHUTILS_H
#define DEEP8_GPUMATHUTILS_H

#include "GPUBasic.h"

#ifdef HAVE_CUDA

namespace Deep8 {
namespace CuMath {

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuExp(const real &in) {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuExp<float>(const float &in) {
	return expf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuExp<double>(const double &in) {
	return exp(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuExp<half>(const half &in) {
	return hexp(in);
}
#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuAbs(const real &in) {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuAbs<float>(const float &in) {
	return fabsf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuAbs<double>(const double &in) {
	return fabs(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuAbs<half>(const half &in) {
	return in >= half(0) ? in : -in;
}
#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuSqrt(const real &in) {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuSqrt<float>(const float &in) {
	return sqrtf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuSqrt<double>(const double &in) {
	return sqrt(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuSqrt<half>(const half &in) {
	return hsqrt(in);
}
#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuLog(const real &in) {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuLog<float>(const float &in) {
	return logf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuLog<double>(const double &in) {
	return log(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuLog<half>(const half &in) {
	return hlog(in);
}
#endif


template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuPow(const real &in, const real &scalar) {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuPow<float>(const float &in, const float &scalar) {
	return powf(in, scalar);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuPow<double>(const double &in, const double &scalar) {
	return pow(in, scalar);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuPow<half>(const half &in, const half &scalar) {
	return __float2half(powf(__half2float(in), __half2float(scalar)));
}
#endif


template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuTanh(const real &in) {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuTanh<float>(const float &in) {
	return tanhf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuTanh<double>(const double &in) {
	return tanh(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuTanh<half>(const half &in) {
	return __float2half(tanh(__half2float(in)));
}
#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuMax(const real &i1, const real &i2) {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuMax<float>(const float &i1, const float &i2) {
	return fmaxf(i1, i2);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuMax<double>(const double &i1, const double &i2) {
	return fmax(i1, i2);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuMax<half>(const half &i1, const half &i2) {
	return i1 >= i2 ? i1 : i2;
}
#endif

template <typename real>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real cuMinValue() {
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cuMinValue() {
	return -FLT_MAX;
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cuMinValue() {
	return -DBL_MAX;
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cuMinValue() {
	return -65504.0;
}
#endif

}




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

//template <unsigned int blockSize, typename real>
//DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE void warpMaxReduce(volatile real *shared, int threadId) {
//	if (blockSize >= 64) { shared[threadId] = cuMax(shared[threadId], shared[threadId + 32]); }
//	if (blockSize >= 32) { shared[threadId] = cuMax(shared[threadId], shared[threadId + 16]); }
//	if (blockSize >= 16) { shared[threadId] = cuMax(shared[threadId], shared[threadId +  8]); }
//	if (blockSize >= 8)  { shared[threadId] = cuMax(shared[threadId], shared[threadId +  4]); }
//	if (blockSize >= 4)  { shared[threadId] = cuMax(shared[threadId], shared[threadId +  2]); }
//	if (blockSize >= 2)  { shared[threadId] = cuMax(shared[threadId], shared[threadId +  1]); }
//}

}

#endif

#endif