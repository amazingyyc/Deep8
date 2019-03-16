cmake_minimum_required(VERSION 3.8)

# if the device have GPU and if build the test
set(HAVE_GPU FALSE)
set(BUILD_PYTHON TRUE)
set(BUILD_TEST TRUE)

set(CMAKE_CXX_STANDARD 11)

if (HAVE_GPU)
	# find the CUDA
	find_package(CUDA)
endif()

if (HAVE_GPU AND CUDA_FOUND)
	# find the CUDA
	message(STATUS "find the CUDA, the version is => ${CUDA_VERSION}, the path is => ${CUDA_TOOLKIT_ROOT_DIR}")
	set(HAVE_CUDA TRUE)

	if (CUDA_HAS_FP16)
	message(STATUS "support half")
	set(HAVE_HALF TRUE)
	endif()

	# include the CUDA head dir
	message(STATUS "include the CUDA head files => ${CUDA_INCLUDE_DIRS}")
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

	#set the CUDA_LIBRARIES to cudart_static, nvlink does not support space in path
	set(CUDA_LIBRARIES cudart_static)

	# include the CUDA Random Number Generation library
	message(STATUS "include the CUDA Random Number Generation library")
    # list(APPEND CUDA_LIBRARIES ${CUDA_curand_LIBRARY})
	list(APPEND CUDA_LIBRARIES curand)

	message(STATUS "include the CUDA Blas library")
	# list(APPEND CUDA_LIBRARIES ${CUDA_cublas_LIBRARY})
	list(APPEND CUDA_LIBRARIES cublas)

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
		# list(APPEND CUDA_LIBRARIES ${CUDNN_LIBRARIES})
		list(APPEND CUDA_LIBRARIES cudnn)
	else()
		message(STATUS "can not find the CUDNN, some Function can not be used in Deep8")
	endif()

	# print the libraries of CUDN or CUDNN
	message(STATUS "the following libraries will be incldue in Deep8 => ${CUDA_LIBRARIES}")
else()
	message(STATUS "can not find the CUDA")
endif()

# set the Maro for Eigen
if (HAVE_GPU AND HAVE_CUDA)
	message(STATUS "find the GPU add set Eigen use multi-threads")
	add_definitions(-DHAVE_CUDA -DEIGEN_USE_THREADS -DEIGEN_FAST_MATH -DEIGEN_NO_CUDA)

	if (HAVE_CUDNN)
		add_definitions(-DHAVE_CUDNN)
	endif()

	if (HAVE_HALF)
		add_definitions(-DHAVE_HALF)
	endif()
else()
	message(STATUS "do not have GPU only support CPU")
	add_definitions(-DEIGEN_USE_THREADS -DEIGEN_FAST_MATH -DEIGEN_NO_CUDA)
endif()

# include
include_directories(third_party)
include_directories(third_party/eigen)
include_directories(deep8/include)
include_directories(deep8/include/basic)
include_directories(deep8/include/math)
include_directories(deep8/include/model)
include_directories(deep8/include/nodes)
include_directories(deep8/include/trainer)
include_directories(deep8/include/utils)

# add deep8
add_subdirectory(deep8)

if (BUILD_PYTHON)
	include_directories(third_party/pybind11/include)
	add_subdirectory(third_party/pybind11)
	add_subdirectory(python)
endif()

if (BUILD_TEST)
	# gtest
	if (MSVC)
		message(STATUS "when use the MSVC set the gtest_force_shared_crt be true to void a compile error")
		set(gtest_force_shared_crt true)
	endif()

    message(STATUS "include googletest")
    add_subdirectory(third_party/googletest)
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
	add_subdirectory(test)
endif()
