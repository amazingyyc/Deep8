cmake_minimum_required(VERSION 3.8)

# if the device have GPU
set(HAVE_GPU true)

if (HAVE_GPU)
	project(Deep8 LANGUAGES CXX CUDA)
else()
    project(Deep8 LANGUAGES CXX)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (HAVE_GPU)
# find the CUDA
find_package(CUDA)
endif()

message(STATUS "========================================================================================================\n")

if (HAVE_GPU AND CUDA_FOUND)
	# find the CUDA
	message(STATUS "find the CUDA, the version is => ${CUDA_VERSION}, the path is => ${CUDA_TOOLKIT_ROOT_DIR}")
	set(HAVE_CUDA TRUE)

	# include the CUDA head dir
	message(STATUS "include the CUDA head files => ${CUDA_INCLUDE_DIRS}")
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

	# include the CUDA Random Number Generation library
	message(STATUS "include the CUDA Random Number Generation library")
    list(APPEND CUDA_LIBRARIES ${CUDA_curand_LIBRARY})

	# begin to find the cudnn
	message(STATUS "begin to find the CUDNN at ENV{CUDNN_ROOT} or {CUDA_TOOLKIT_ROOT_DIR}")

	if (DEFINED ENV{CUDNN_ROOT})
		set(CUDNN_ROOT $ENV{CUDNN_ROOT})
	endif()

	if (NOT DEFINED CUDNN_ROOT)
		set(CUDNN_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
	endif()

	# find the CUDNN head file path
	find_path(CUDNN_INCLUDE_DIRS cudnn.h HINTS ${CUDNN_ROOT} ${CUDNN_ROOT}/include)

	# find the library
	find_library(CUDNN_LIBRARIES NAMES libcudnn.so cudnn.lib PATHS ${CUDNN_ROOT} ${CUDNN_ROOT}/lib ${CUDNN_ROOT}/lib64 ${CUDNN_ROOT}/lib/x64)

	if (CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
		message(STATUS "find the CUDNN include dirs is => ${CUDNN_INCLUDE_DIRS}, library dir is => ${CUDNN_LIBRARIES}")
		set(HAVE_CUDNN TRUE)

		include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
		list(APPEND CUDA_LIBRARIES ${CUDNN_LIBRARIES})
	else()
		message(STATUS "can not find the CUDNN, some Function can not be used in Deep8")
	endif()

	# print the libraries of CUDN or CUDNN
	message(STATUS "the following libraries will be incldue in Deep8 => ${CUDA_LIBRARIES}")
else()
	message(STATUS "can not find the CUDA")
endif()

# fix for MSVC
# if(MSVC)
#	message(STATUS "set /bigobj to avoid compile error")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /Gy")
# endif()

# set the Maro for Eigen Compile
if (HAVE_GPU AND HAVE_CUDA)
	message(STATUS "find the GPU add set Eigen use multi-threads")
	add_definitions(-DHAVE_CUDA -DEIGEN_USE_THREADS -DEIGEN_FAST_MATH)
	# add_definitions(-DHAVE_CUDA -DEIGEN_USE_THREADS)
	
	if (HAVE_CUDNN)
		add_definitions(-DHAVE_CUDNN)
	endif()
else()
	message(STATUS "do not have GPU only support CPU")
	add_definitions(-DEIGEN_USE_THREADS -DEIGEN_FAST_MATH)
endif()

# include the Eigen
include_directories(third_party/eigen)

# gtest
if (MSVC)
	message(STATUS "when use the MSVC set the gtest_force_shared_crt be true to void a compile error")
	set(gtest_force_shared_crt true)
endif()

message(STATUS "include googletest")
add_subdirectory(third_party/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# deep8
include_directories(deep8)
include_directories(deep8/basic)
include_directories(deep8/model)
include_directories(deep8/nodes)
include_directories(deep8/utils)

set ( DEEP8_HEAD_FILES

		deep8/Deep8.h

		deep8/basic/Device.h
		deep8/basic/Exception.h
		deep8/basic/MemoryAllocator.h
		deep8/basic/MemoryPool.h
        deep8/basic/Shape.h
		deep8/basic/TensorStorage.h
		deep8/basic/Tensor.h
		deep8/basic/CudaException.h

		deep8/model/Executor.h
		deep8/model/DefaultExecutor.h
		deep8/model/Expression.h
		deep8/model/Trainer.h
		deep8/model/TensorInit.h

		deep8/nodes/Abs.h
		deep8/nodes/Add.h
		deep8/nodes/AvgPooling2d.h
		deep8/nodes/ConstantParameter.h
		deep8/nodes/Conv2d.h
		deep8/nodes/DeConv2d.h
		deep8/nodes/Divide.h
		deep8/nodes/Exp.h
		deep8/nodes/Function.h
		deep8/nodes/InputParameter.h
		deep8/nodes/L1Norm.h
		deep8/nodes/L2Norm.h
		deep8/nodes/Linear.h
		deep8/nodes/Log.h
		deep8/nodes/LReLu.h
		deep8/nodes/MatrixMultiply.h
		deep8/nodes/MaxPooling2d.h
		deep8/nodes/Minus.h
		deep8/nodes/Multiply.h
		deep8/nodes/Node.h
		deep8/nodes/Parameter.h
		deep8/nodes/Pow.h
		deep8/nodes/ReLu.h
		deep8/nodes/ReShape.h
		deep8/nodes/Sigmoid.h
		deep8/nodes/Softmax.h
		deep8/nodes/Square.h
		deep8/nodes/SumElements.h
		deep8/nodes/TanH.h
		deep8/nodes/Variable.h

		deep8/utils/CudaMathUtils.h
		deep8/utils/MathUtils.h
		deep8/utils/ShapeUtils.h
		deep8/utils/TensorUtils.h
		deep8/utils/Utils.h
		deep8/model/Executor.h )

set ( DEEP8_SRC_FILES

		deep8/basic/TensorStorage.cpp
		deep8/basic/Tensor.cpp

		deep8/model/Executor.cpp
		deep8/model/DefaultExecutor.cpp
		deep8/model/Trainer.cpp

		deep8/nodes/Function.cpp
		deep8/nodes/Abs.cpp
		deep8/nodes/Variable.cpp
		deep8/nodes/Parameter.cpp
		deep8/nodes/InputParameter.cpp
		deep8/nodes/ConstantParameter.cpp
	
		# deep8/basic/MemoryAllocator.cpp
		# deep8/basic/TensorStorage.cpp
		# deep8/basic/Tensor.cpp
		# 
		# deep8/nodes/Abs.cpp 
		# deep8/nodes/Add.cpp 
		# deep8/nodes/AddScalar.cpp 
		# deep8/nodes/AvgPooling2d.cpp 
		# deep8/nodes/ConstantParameter.cpp 
		# deep8/nodes/Conv2d.cpp 
		# deep8/nodes/DeConv2d.cpp 
		# deep8/nodes/Divide.cpp 
		# deep8/nodes/DivideScalar.cpp 
		# deep8/nodes/Exp.cpp 
		# deep8/nodes/Function.cpp 
		# deep8/nodes/InputParameter.cpp 
		# deep8/nodes/L1Norm.cpp 
		# deep8/nodes/L2Norm.cpp 
		# deep8/nodes/Linear.cpp 
		# deep8/nodes/Log.cpp 
		# deep8/nodes/LReLu.cpp 
		# deep8/nodes/MatrixMultiply.cpp 
		# deep8/nodes/MaxPooling2d.cpp 
		# deep8/nodes/Minus.cpp 
		# deep8/nodes/MinusScalar.cpp 
		# deep8/nodes/Multiply.cpp 
		# deep8/nodes/MultiplyScalar.cpp 
		# deep8/nodes/Parameter.cpp 
		# deep8/nodes/Pow.cpp 
		# deep8/nodes/ReLu.cpp 
		# deep8/nodes/ReShape.cpp 
		# deep8/nodes/ScalarDivide.cpp 
		# deep8/nodes/ScalarMinus.cpp 
		# deep8/nodes/Sigmoid.cpp 
		# deep8/nodes/Softmax.cpp 
		# deep8/nodes/Square.cpp 
		# deep8/nodes/SumElements.cpp 
		# deep8/nodes/TanH.cpp 
		# deep8/nodes/Variable.cpp 
		)

set ( DEEP8_CUDA_SRC_FILES

		deep8/model/Executor.cu
		deep8/model/Trainer.cu

		deep8/nodes/Abs.cu

		# deep8/basic/MemoryAllocator.cu
		# deep8/basic/TensorStorage.cu
		# deep8/basic/Tensor.cu
		# 
		# deep8/nodes/L1Norm.cu
		)

# Deep8 test
include_directories(test)

set ( DEEP8_TEST_HEAD_FILES
		test/AbsTest.h
		test/AddScalarTest.h
		test/AddTest.h
		test/AvgPooling2dTest.h
		test/Conv2dTest.h
		test/CPUMemoryPoolTest.h
		test/GPUMemoryPoolTest.h
		test/DeConv2dTest.h
		test/DivideScalarTest.h
		test/DivideTest.h
		test/EigenTest.h
		test/ExpTest.h
		test/L1NormTest.h
		test/L2NormTest.h
		test/LinearRegressionTest.h
		test/LinearTest.h
		test/LogTest.h
		test/LReLuTest.h
		test/MatrixMultiplyTest.h
		test/MaxPooling2dTest.h
		test/MinusScalarTest.h
		test/MinusTest.h
		test/MultiplyScalarTest.h
		test/MultiplyTest.h
		test/PowTest.h
		test/ReLuTest.h
		test/SigmoidTest.h
		test/SoftmaxTest.h
		test/SquareTest.h
		test/SumElementsTest.h
		test/TanHTest.h
		test/TestUtils.h
        test/LinearRegressionHalfTest.h)

set (DEEP8_TEST_SRC_FILES
		test/Deep8Test.cpp)

set (DEEP8_TEST_CUDA_SRC_FILES
		test/Deep8Test.cpp)


if (HAVE_GPU AND HAVE_CUDA)
	message(STATUS "compile with the CUDA")

	# enable separable compilation
	set(CUDA_SEPARABLE_COMPILATION ON)

	# eigen enabel GPU
	#if(CUDA_ARCH)
	#	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${CUDA_ARCH};-std=c++11;-DVERBOSE;-DEIGEN_USE_THREADS;-DHAVE_CUDA;-D_FORCE_INLINES;")
	#else()
	#    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61;-std=c++11;-DVERBOSE;-DEIGEN_USE_THREADS;-DHAVE_CUDA;-D_FORCE_INLINES;")
	#	message(STATUS ${CUDA_NVCC_FLAGS})
	#endif()

	
	string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70  -gencode arch=compute_72,code=sm_72 -DVERBOSE -DEIGEN_USE_THREADS -DHAVE_CUDA -D_FORCE_INLINES")

	# enable cudnn
	if (HAVE_CUDNN)
		string(APPEND CMAKE_CUDA_FLAGS "-DHAVE_CUDNN")
	endif()

	# add_executable(Deep8 ${DEEP8_CUDA_SRC_FILES} ${DEEP8_SRC_FILES} ${DEEP8_TEST_CUDA_SRC_FILES} ${DEEP8_HEAD_FILES} ${DEEP8_TEST_HEAD_FILES})
	add_executable(Deep8 ${DEEP8_CUDA_SRC_FILES} ${DEEP8_SRC_FILES} ${DEEP8_TEST_CUDA_SRC_FILES})

	# add cuBlas library
	cuda_add_cublas_to_target(Deep8)

	# link the gtest and CUDA library
	target_link_libraries(Deep8 gtest ${CUDA_LIBRARIES})
else()
	message(STATUS "can not find CUDA only compile for CPU")

	add_executable(Deep8 ${DEEP8_SRC_FILES} ${DEEP8_TEST_SRC_FILES} ${DEEP8_HEAD_FILES} ${DEEP8_TEST_HEAD_FILES})
	target_link_libraries(Deep8 gtest)

endif()

message(STATUS "\n========================================================================================================")
