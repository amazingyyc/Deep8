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

file(GLOB DEEP8_HEAD_FILES "include/*.h")
file(GLOB DEEP8_CUDA_HEAD_FILES "include/*.cuh")

set(DEEP8_SRC_FILES

		src/model/AutoBatchCodeHelper.cpp
		src/model/Executor.cpp
		src/model/EagerExecutor.cpp
		src/model/LazyExecutor.cpp
		src/model/LearningRateIterator.cpp
		src/model/Device.cpp
		src/model/MemoryAllocator.cpp
		src/model/MemoryPool.cpp
		src/model/OutputEntry.cpp
		src/model/Shape.cpp
		src/model/TensorInit.cpp
		src/model/TensorStorage.cpp
		src/model/Tensor.cpp
		src/model/Trainer.cpp

		src/nodes/Abs.cpp
		src/nodes/Add.cpp
		src/nodes/AddScalar.cpp
		src/nodes/AvgPooling2d.cpp
		src/nodes/Batch.cpp
		src/nodes/Conv2d.cpp
		src/nodes/CrossEntropy.cpp
		src/nodes/DeConv2d.cpp
		src/nodes/Divide.cpp
		src/nodes/DivideScalar.cpp
		src/nodes/Exp.cpp
		src/nodes/Function.cpp
		src/nodes/L1Norm.cpp
		src/nodes/L2Norm.cpp
		src/nodes/Linear.cpp
		src/nodes/Log.cpp
		src/nodes/LogSoftmax.cpp
		src/nodes/LReLu.cpp
		src/nodes/MatrixMultiply.cpp
		src/nodes/MaxPooling2d.cpp
		src/nodes/Minus.cpp
		src/nodes/MinusScalar.cpp
		src/nodes/Multiply.cpp
		src/nodes/MultiplyScalar.cpp
		src/nodes/Node.cpp
		src/nodes/Parameter.cpp
		src/nodes/ReduceSum.cpp
		src/nodes/ReLu.cpp
		src/nodes/ReShape.cpp
		src/nodes/ScalarDivide.cpp
		src/nodes/ScalarMinus.cpp
		src/nodes/Sigmoid.cpp
		src/nodes/Softmax.cpp
		src/nodes/Square.cpp
		src/nodes/Tanh.cpp
		src/nodes/UnBatch.cpp
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
		src/nodes/Batch.cu
		src/nodes/Conv2d.cu
		src/nodes/CrossEntropy.cu
		src/nodes/DeConv2d.cu
		src/nodes/Divide.cu
		src/nodes/DivideScalar.cu
		src/nodes/Exp.cu
		src/nodes/L1Norm.cu
		src/nodes/L2Norm.cu
		src/nodes/Linear.cu
		src/nodes/Log.cu
		src/nodes/LogSoftmax.cu
		src/nodes/LReLu.cu
		src/nodes/MatrixMultiply.cu
		src/nodes/MaxPooling2d.cu
		src/nodes/Minus.cu
		src/nodes/MinusScalar.cu
		src/nodes/Multiply.cu
		src/nodes/MultiplyScalar.cu
		src/nodes/ReduceSum.cu
		src/nodes/ReLu.cu
		src/nodes/ScalarDivide.cu
		src/nodes/ScalarMinus.cu
		src/nodes/Sigmoid.cu
		src/nodes/Softmax.cu
		src/nodes/Square.cu
		src/nodes/Tanh.cu
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

	add_library(deep8_native ${DEEP8_SRC_FILES} ${DEEP8_CUDA_SRC_FILES} ${DEEP8_HEAD_FILES} ${DEEP8_CUDA_HEAD_FILES})

	# link the gtest and CUDA library
	target_link_libraries(deep8_native ${CUDA_LIBRARIES})
else()
	add_library(deep8_native ${DEEP8_SRC_FILES} ${DEEP8_HEAD_FILES})
endif()

install(TARGETS deep8_native DESTINATION lib)
install(FILES ${DEEP8_HEAD_FILES} DESTINATION include/deep8)

message(STATUS "========================================================================================================")
message(STATUS "end compile deep8_native")
message(STATUS "========================================================================================================")