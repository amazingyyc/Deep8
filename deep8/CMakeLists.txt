if (HAVE_GPU)
	project(deep8 LANGUAGES CXX CUDA)
else()
    project(deep8 LANGUAGES CXX)
endif()

message(STATUS "========================================================================================================")
message(STATUS "Begin compile deep8")
message(STATUS "========================================================================================================")

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
	add_definitions(-DEIGEN_USE_THREADS -DEIGEN_FAST_MATH)
endif()

# deep8 include
include_directories(include)

file(GLOB DEEP8_HEAD_FILES "include/*.h")

set(DEEP8_SRC_FILES

		src/model/Executor.cpp
		src/model/EagerExecutor.cpp
		src/model/Device.cpp
		src/model/MemoryAllocator.cpp
		src/model/MemoryPool.cpp
		src/model/Shape.cpp
		src/model/TensorInit.cpp
		src/model/TensorStorage.cpp
		src/model/Tensor.cpp
		src/model/Trainer.cpp

		src/nodes/Abs.cpp
		src/nodes/Add.cpp
		src/nodes/AddScalar.cpp
		src/nodes/AvgPooling2d.cpp
		src/nodes/Conv2d.cpp
		src/nodes/DeConv2d.cpp
		src/nodes/Divide.cpp
		src/nodes/DivideScalar.cpp
		src/nodes/Exp.cpp
		src/nodes/Function.cpp
		src/nodes/InputParameter.cpp
		src/nodes/L1Norm.cpp
		src/nodes/L2Norm.cpp
		src/nodes/Linear.cpp
		src/nodes/Log.cpp
		src/nodes/LReLu.cpp
		src/nodes/MatrixMultiply.cpp
		src/nodes/MaxPooling2d.cpp
		src/nodes/Minus.cpp
		src/nodes/MinusScalar.cpp
		src/nodes/Multiply.cpp
		src/nodes/MultiplyScalar.cpp
		src/nodes/Parameter.cpp
		src/nodes/Pow.cpp
		src/nodes/ReLu.cpp
		src/nodes/ReShape.cpp
		src/nodes/ScalarDivide.cpp
		src/nodes/ScalarMinus.cpp
		src/nodes/Sigmoid.cpp
		src/nodes/Softmax.cpp
		src/nodes/Square.cpp
		src/nodes/SumElements.cpp
		src/nodes/TanH.cpp
		src/nodes/Variable.cpp

		)

set(DEEP8_CUDA_SRC_FILES

		src/model/Executor.cu
		src/model/GPUDevice.cu
		src/model/GPUMemoryAllocator.cu
		src/model/GPUMemoryPool.cu
		src/model/TensorInit.cu
		src/model/Trainer.cu
		
		src/nodes/Abs.cu
		src/nodes/Add.cu
		src/nodes/AddScalar.cu
		src/nodes/AvgPooling2d.cu
		src/nodes/Conv2d.cu
		src/nodes/DeConv2d.cu
		src/nodes/Divide.cu
		src/nodes/DivideScalar.cu
		src/nodes/Exp.cu
		src/nodes/L1Norm.cu
		src/nodes/L2Norm.cu
		src/nodes/Linear.cu
		src/nodes/Log.cu
		src/nodes/LReLu.cu
		src/nodes/MatrixMultiply.cu
		src/nodes/MaxPooling2d.cu
		src/nodes/Minus.cu
		src/nodes/MinusScalar.cu
		src/nodes/Multiply.cu
		src/nodes/MultiplyScalar.cu
		src/nodes/Pow.cu
		src/nodes/ReLu.cu
		src/nodes/ScalarDivide.cu
		src/nodes/ScalarMinus.cu
		src/nodes/Sigmoid.cu
		src/nodes/Softmax.cu
		src/nodes/Square.cu
		src/nodes/SumElements.cu
		src/nodes/TanH.cu
		)

if (HAVE_GPU AND HAVE_CUDA)
	# enable separable compilation
	set(CUDA_SEPARABLE_COMPILATION ON)
	
	string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70  -gencode arch=compute_72,code=sm_72 -DVERBOSE -DEIGEN_USE_THREADS  -DEIGEN_FAST_MATH -DHAVE_CUDA -DEIGEN_NO_CUDA -D_FORCE_INLINES")

	# enable cudnn
	if (HAVE_CUDNN)
		string(APPEND CMAKE_CUDA_FLAGS "-DHAVE_CUDNN")
	endif()

	if (HAVE_HALF)
		string(APPEND CMAKE_CUDA_FLAGS "-DHAVE_HALF")
	endif()

	add_library(deep8 ${DEEP8_SRC_FILES} ${DEEP8_CUDA_SRC_FILES} ${DEEP8_HEAD_FILES})

	# add cuBlas library
	cuda_add_cublas_to_target(deep8)

	# link the gtest and CUDA library
	target_link_libraries(deep8 gtest ${CUDA_LIBRARIES})
else()
	add_library(deep8  ${DEEP8_SRC_FILES} ${DEEP8_HEAD_FILES})
endif()

install(TARGETS deep8 DESTINATION lib)
install(FILES ${DEEP8_HEAD_FILES} DESTINATION include/deep8)

message(STATUS "========================================================================================================")
message(STATUS "end compile deep8")
message(STATUS "========================================================================================================")