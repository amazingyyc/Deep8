#ifndef DEEP8_GPUEXCEPTION_H
#define DEEP8_GPUEXCEPTION_H

#include "Exception.h"
#include "GPUBasic.h"

namespace Deep8 {

#ifdef HAVE_CUDA

#define CUDA_CHECK(cudaExecute)							\
{														\
	auto ret = cudaExecute;								\
	if (ret != cudaSuccess) {							\
		DEEP8_RUNTIME_ERROR("the CUDA get a error: "	\
		<< #cudaExecute									\
		<< ","											\
	    << cudaGetErrorString(ret));				    \
	}													\
};														\

#define CUBLAS_CHECK(cudaExecute)						 \
{														 \
	auto ret = cudaExecute;								 \
	if (ret != CUBLAS_STATUS_SUCCESS) {					 \
		DEEP8_RUNTIME_ERROR("the cuBlas get a error: "	 \
			<< #cudaExecute								 \
			<< ".");								     \
	}													 \
};														 \

#define CURAND_CHECK(cudaExecute)						 \
{														 \
	auto ret = cudaExecute;								 \
	if (ret != CURAND_STATUS_SUCCESS) {					 \
		DEEP8_RUNTIME_ERROR("the cuRand get a error: "	 \
			<< #cudaExecute								 \
			<< ".");								     \
	}													 \
};														 \

#ifdef HAVE_CUDNN

#define CUDNN_CHECK(cudnnExecute)						 \
{														 \
	auto ret = cudnnExecute;							 \
	if (ret != CUDNN_STATUS_SUCCESS) {					 \
		DEEP8_RUNTIME_ERROR("the cudnn get a error:"	 \
			<< #cudnnExecute							 \
			<< ","										 \
		    << cudnnGetErrorString(ret));				 \
	}													 \
}														 \

#endif

#endif

}

#endif