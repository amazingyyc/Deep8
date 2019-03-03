#ifndef DEEP8_MATH_GPUMATH_H
#define DEEP8_MATH_GPUMATH_H

#include "GPUBasic.h"

namespace Deep8 {
namespace Math {

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaExp(const T &in) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaExp<float>(const float &in) {
	return expf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaExp<double>(const double &in) {
	return exp(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaExp<half>(const half &in) {
	return hexp(in);
}
#endif

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaAbs(const T &in) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaAbs<float>(const float &in) {
	return fabsf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaAbs<double>(const double &in) {
	return fabs(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaAbs<half>(const half &in) {
	return in >= half(0) ? in : -in;
}
#endif

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaSqrt(const T &in) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaSqrt<float>(const float &in) {
	return sqrtf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaSqrt<double>(const double &in) {
	return sqrt(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaSqrt<half>(const half &in) {
	return hsqrt(in);
}
#endif

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaLog(const T &in) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaLog<float>(const float &in) {
	return logf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaLog<double>(const double &in) {
	return log(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaLog<half>(const half &in) {
	return hlog(in);
}
#endif


template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaPow(const T &in, const T &scalar) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaPow<float>(const float &in, const float &scalar) {
	return powf(in, scalar);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaPow<double>(const double &in, const double &scalar) {
	return pow(in, scalar);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaPow<half>(const half &in, const half &scalar) {
	return __float2half(powf(__half2float(in), __half2float(scalar)));
}
#endif

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaTanh(const T &in) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaTanh<float>(const float &in) {
	return tanhf(in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaTanh<double>(const double &in) {
	return tanh(in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaTanh<half>(const half &in) {
	return __float2half(tanh(__half2float(in)));
}
#endif

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaSigmoid(const T &in) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaSigmoid<float>(const float &in) {
	return 0.5 + 0.5 * tanhf(0.5 * in);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaSigmoid<double>(const double &in) {
	return 0.5 + 0.5 * tanh(0.5 * in);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaSigmoid<half>(const half &in) {
	return __float2half(cudaSigmoid<float>(__half2float(in)));
}
#endif

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaMax(const T &i1, const T &i2) {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaMax<float>(const float &i1, const float &i2) {
	return fmaxf(i1, i2);
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaMax<double>(const double &i1, const double &i2) {
	return fmax(i1, i2);
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaMax<half>(const half &i1, const half &i2) {
	return i1 >= i2 ? i1 : i2;
}
#endif

template <typename T>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T cudaMinValue() {
	DEEP8_RUNTIME_ERROR("the type is not support");
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float cudaMinValue() {
	return -FLT_MAX;
}

template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double cudaMinValue() {
	return -DBL_MAX;
}

#ifdef HAVE_HALF
template <>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half cudaMinValue() {
	return -65504.0;
}
#endif

}
}

#endif