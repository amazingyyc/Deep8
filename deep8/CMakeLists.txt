if (BUILD_WITH_GPU)
    project(deep8_native LANGUAGES CXX CUDA)
else()
    project(deep8_native LANGUAGES CXX)
endif()

message(STATUS "Begin compile deep8_native")

# c++11
set(CMAKE_CXX_STANDARD 11)

if (BUILD_WITH_GPU)
    if (NOT CUDA_FOUND)
        # can not find the GPU
        message(FATAL_ERROR, "can not find the CUDA, set BUILD_WITH_GPU to be FALSE to compiler")
    endif()

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
    message(STATUS "include the CUDA cudart_static")
    set(CUDA_LIBRARIES cudart_static)

    message(STATUS "include the CUDA Random Number Generation library")
    list(APPEND CUDA_LIBRARIES curand)

    message(STATUS "include the CUDA Blas library")
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
        list(APPEND CUDA_LIBRARIES cudnn)
    else()
        message(WARNING "can not find the CUDNN, some Function can not be used in Deep8")
    endif()

    # print the libraries of CUDN or CUDNN
    message(STATUS "the following libraries will be incldue in Deep8 => ${CUDA_LIBRARIES}")
endif()

if (BUILD_WITH_GPU)
    message(STATUS "build with GPU and set Eigen use multi-threads")
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

if (BUILD_WITH_GPU)
    # enable separable compilation
    set(CUDA_SEPARABLE_COMPILATION ON)
    
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70  -gencode arch=compute_72,code=sm_72 -DVERBOSE -DEIGEN_USE_THREADS -DEIGEN_FAST_MATH -DHAVE_CUDA -DEIGEN_NO_CUDA -D_FORCE_INLINES")

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
install(DIRECTORY deep8/include/ DESTINATION include/deep8)

message(STATUS "end compile deep8_native")