#ifndef DEEP8_BASIC_H
#define DEEP8_BASIC_H

#ifdef HAVE_CUDA

#define DEEP8_CUDA_FUNC __device__

#if _MSC_VER || __INTEL_COMPILER
#define DEEP8_CUDA_INLINE __forceinline
#else
#define DEEP8_CUDA_INLINE inline
#endif
#endif

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cstddef>
#include <queue>
#include <random>
#include <utility>
#include <typeinfo>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#ifdef __GUNC__
#include <mm_malloc.h>
#include <zconf.h>
#endif

#ifdef HAVE_HALF
#include <cuda_fp16.h>
#endif

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**define the byte type*/
typedef unsigned char byte;

#ifdef HAVE_HALF
#define DEEP8_DECLARATION_INSTANCE(name)    \
            template class name<float>;     \
            template class name<double>;    \
            template class name<half>;
#else
#define DEEP8_DECLARATION_INSTANCE(name)    \
            template class name<float>;     \
            template class name<double>;
#endif


#ifdef HAVE_HALF
#define DEEP8_DECLARATION_GPU_FUNC(name)	\
template void name<half>::forwardGPU(const std::vector<const Tensor<half>*> &, Tensor<half> *);			\
template void name<float>::forwardGPU(const std::vector<const Tensor<float>*> &, Tensor<float> *);		\
template void name<double>::forwardGPU(const std::vector<const Tensor<double>*> &, Tensor<double> *);	\
template void name<half>::backwardGPU(const std::vector<const Tensor<half>*>&, const Tensor<half>*, const Tensor<half>*, size_t, Tensor<half>*);			\
template void name<float>::backwardGPU(const std::vector<const Tensor<float>*>&, const Tensor<float>*, const Tensor<float>*, size_t, Tensor<float>*);		\
template void name<double>::backwardGPU(const std::vector<const Tensor<double>*>&, const Tensor<double>*, const Tensor<double>*, size_t, Tensor<double>*);
#else
#define DEEP8_DECLARATION_GPU_FUNC(name)	\
template void name<float>::forwardGPU(const std::vector<const Tensor<float>*> &inputs, Tensor<float> *output);		\
template void name<double>::forwardGPU(const std::vector<const Tensor<double>*> &inputs, Tensor<double> *output);		\
template void name<float>::backwardGPU(const std::vector<const Tensor<float>*> &inputs, const Tensor<float> *output, const Tensor<float> *outputGradient, size_t index, Tensor<float> *iGradient);	\
template void name<double>::backwardGPU(const std::vector<const Tensor<double>*> &inputs, const Tensor<double> *output, const Tensor<double> *outputGradient, size_t index, Tensor<double> *iGradient);
#endif

#ifdef HAVE_HALF
#define DEEP8_RE_DECLARATION_HALF_FUNC(name)										\
	template <>																		\
	void name<half>::forwardCPU(const std::vector<const Tensor<half>*> &inputs,		\
					                                    Tensor<half> *output) {		\
		DEEP8_RUNTIME_ERROR("CPU not support half");								\
	}																				\
	template <>																		\
	void name<half>::backwardCPU(const std::vector<const Tensor<half>*> &inputs,	\
								 const Tensor<half> *output,						\
								 const Tensor<half> *outputGradient,				\
								 size_t index,										\
								 Tensor<half> *iGradient) {							\
		DEEP8_RUNTIME_ERROR("CPU not support half");								\
	}
#else
#define DEEP8_RE_DECLARATION_HALF_FUNC(name)
#endif

#endif