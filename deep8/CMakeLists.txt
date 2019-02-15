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

# file(GLOB DEEP8_HEAD_FILES "include/*.h")
# file(GLOB DEEP8_CUDA_HEAD_FILES "include/*.cuh")

set(DEEP8_SRC_FILES
		src/math/Abs.cpp
		src/math/Add.cpp
		src/math/AvgPooling2d.cpp
		src/math/Conv2d.cpp
		src/math/CrossEntropy.cpp
		src/math/DeConv2d.cpp
		src/math/Divide.cpp
		src/math/Exp.cpp
		src/math/L1Norm.cpp
		src/math/L2Norm.cpp
		src/math/Linear.cpp
		src/math/Log.cpp
		src/math/LogSoftmax.cpp
		src/math/LReLu.cpp
		src/math/MatrixMultiply.cpp
		src/math/MaxPooling2d.cpp
		src/math/Minus.cpp
		src/math/Multiply.cpp
		src/math/ReduceMean.cpp
		src/math/ReduceSum.cpp
		src/math/ReLu.cpp
		src/math/Sigmoid.cpp
		src/math/Softmax.cpp
		src/math/Square.cpp
		src/math/Tanh.cpp

		src/model/AutoBatchCodeHelper.cpp
		src/model/Device.cpp
		#src/model/EagerExecutor.cpp
		#src/model/Executor.cpp
		#src/model/LazyExecutor.cpp
		#src/model/LearningRateIterator.cpp
		src/model/MemoryAllocator.cpp
		src/model/MemoryPool.cpp
		src/model/OutputEntry.cpp
		src/model/Shape.cpp
		src/model/Tensor.cpp
		#src/model/TensorInit.cpp
		src/model/TensorStorage.cpp
		
		src/nodes/Abs.cpp
		src/nodes/Add.cpp
		src/nodes/AvgPooling2d.cpp
		src/nodes/Batch.cpp
		src/nodes/Conv2d.cpp
		src/nodes/CrossEntropy.cpp
		src/nodes/DeConv2d.cpp
		src/nodes/Divide.cpp
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
		src/nodes/Multiply.cpp
		src/nodes/Node.cpp
		src/nodes/Parameter.cpp
		src/nodes/ReduceMean.cpp
		src/nodes/ReduceSum.cpp
		src/nodes/ReLu.cpp
		src/nodes/ReShape.cpp
		src/nodes/Sigmoid.cpp
		src/nodes/Softmax.cpp
		src/nodes/Square.cpp
		src/nodes/Tanh.cpp
		src/nodes/UnBatch.cpp
		src/nodes/Variable.cpp

		#src/trainer/Trainer.cpp
		)

set(DEEP8_CUDA_SRC_FILES

		src/math/Abs.cu
		src/math/Add.cu
		src/math/AvgPooling2d.cu
		src/math/Conv2d.cu
		src/math/CrossEntropy.cu
		src/math/DeConv2d.cu
		src/math/Divide.cu
		src/math/Exp.cu
		src/math/L1Norm.cu
		src/math/L2Norm.cu
		src/math/Linear.cu
		src/math/Log.cu
		src/math/LogSoftmax.cu
		src/math/LReLu.cu
		src/math/MatrixMultiply.cu
		src/math/MaxPooling2d.cu
		src/math/Minus.cu
		src/math/Multiply.cu
		src/math/ReduceMean.cu
		src/math/ReduceSum.cu
		src/math/ReLu.cu
		src/math/Sigmoid.cu
		src/math/Softmax.cu
		src/math/Square.cu
		src/math/Tanh.cu

		#src/model/Executor.cu
		src/model/GPUDevice.cu
		src/model/GPUMemoryAllocator.cu
		src/model/GPUMemoryPool.cu
		#src/model/TensorInit.cu

		#src/trainer/Trainer.cu
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

	# add_library(deep8_native ${DEEP8_SRC_FILES} ${DEEP8_CUDA_SRC_FILES} ${DEEP8_HEAD_FILES} ${DEEP8_CUDA_HEAD_FILES})
	add_library(deep8_native ${DEEP8_SRC_FILES} ${DEEP8_CUDA_SRC_FILES})

	# link the gtest and CUDA library
	target_link_libraries(deep8_native ${CUDA_LIBRARIES})
else()
	#add_library(deep8_native ${DEEP8_SRC_FILES} ${DEEP8_HEAD_FILES})
	add_library(deep8_native ${DEEP8_SRC_FILES} )
endif()

#install(TARGETS deep8_native DESTINATION lib)
#install(FILES ${DEEP8_HEAD_FILES} DESTINATION include/deep8)

message(STATUS "========================================================================================================")
message(STATUS "end compile deep8_native")
message(STATUS "========================================================================================================")