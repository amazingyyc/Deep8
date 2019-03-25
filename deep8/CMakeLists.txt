if (HAVE_GPU)
	project(deep8_native LANGUAGES CXX CUDA)
else()
    project(deep8_native LANGUAGES CXX)
endif()

message(STATUS "========================================================================================================")
message(STATUS "Begin compile deep8_native")
message(STATUS "========================================================================================================")

# deep8 include
include_directories(include)
include_directories(include/basic)
include_directories(include/math)
include_directories(include/model)
include_directories(include/nodes)
include_directories(include/trainer)
include_directories(include/utils)

file(GLOB_RECURSE DEEP8_HEAD_FILES "include/*.h")
file(GLOB_RECURSE DEEP8_SRC_FILES "src/*cpp")
file(GLOB_RECURSE DEEP8_CUDA_SRC_FILES "src/*cu")

if (HAVE_GPU AND HAVE_CUDA)
	# enable separable compilation
	set(CUDA_SEPARABLE_COMPILATION ON)
	
	string(APPEND CMAKE_CUDA_FLAGS " -std=c++11 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70  -gencode arch=compute_72,code=sm_72 -DVERBOSE -DEIGEN_USE_THREADS  -DEIGEN_FAST_MATH -DHAVE_CUDA -DEIGEN_NO_CUDA -D_FORCE_INLINES")

	# enable cudnn
	if (HAVE_CUDNN)
		string(APPEND CMAKE_CUDA_FLAGS "-DHAVE_CUDNN")
	endif()

	if (HAVE_HALF)
		string(APPEND CMAKE_CUDA_FLAGS "-DHAVE_HALF")
	endif()

	add_library(deep8_native ${DEEP8_SRC_FILES} ${DEEP8_CUDA_SRC_FILES} ${DEEP8_HEAD_FILES})

	# link the gtest and CUDA library
	target_link_libraries(deep8_native ${CUDA_LIBRARIES})
else()
	add_library(deep8_native ${DEEP8_SRC_FILES} ${DEEP8_HEAD_FILES})
endif()

install(TARGETS deep8_native DESTINATION lib)
# install(FILES ${DEEP8_HEAD_FILES} DESTINATION include/deep8)
install(DIRECTORY deep8/include/ DESTINATION include/deep8)

message(STATUS "========================================================================================================")
message(STATUS "end compile deep8_native")
message(STATUS "========================================================================================================")