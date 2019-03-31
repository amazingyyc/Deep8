if (BUILD_WITH_GPU)
	project(deep8_test LANGUAGES CXX CUDA)
else()
    project(deep8_test LANGUAGES CXX)
endif()

# c++11
set(CMAKE_CXX_STANDARD 11)

if (BUILD_WITH_GPU)
    add_definitions(-DHAVE_CUDA)

	set(DEEP8_TEST_SRC_FILES Deep8Test.cu)
else()
	set(DEEP8_TEST_SRC_FILES Deep8Test.cpp)
endif()

add_definitions(-DEIGEN_USE_THREADS -DEIGEN_FAST_MATH -DEIGEN_NO_CUDA)

add_executable(deep8_test ${DEEP8_TEST_SRC_FILES})
target_link_libraries(deep8_test deep8_native gtest)
