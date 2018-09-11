#ifndef DEEP8_MACRO_H
#define DEEP8_MACRO_H

#ifdef HAVE_CUDA

#define DEEP8_CUDA_FUNC __host__ __device__

#if _MSC_VER || __INTEL_COMPILER
#define DEEP8_CUDA_INLINE __forceinline
#else
#define DEEP8_CUDA_INLINE inline
#endif

#endif // HAVE_CUDA


#endif