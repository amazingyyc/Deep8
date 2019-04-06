cmake_minimum_required(VERSION 3.8)

# if the device have GPU and if build the test
set(BUILD_WITH_GPU FALSE)
set(BUILD_PYTHON TRUE)
set(BUILD_TEST FALSE)

if (BUILD_WITH_GPU)
    # find the CUDA
    find_package(CUDA)
endif()

# include third_party
include_directories(third_party)
include_directories(third_party/eigen)

# add deep8
include_directories(deep8/include)
add_subdirectory(deep8)

# add python module
if (BUILD_PYTHON)
	include_directories(third_party/pybind11/include)
	add_subdirectory(third_party/pybind11)
	add_subdirectory(python)
endif()

# add test
if (BUILD_TEST)
	if (MSVC)
		message(STATUS "when use the MSVC set the gtest_force_shared_crt be true to void a compile error")
		set(gtest_force_shared_crt true)
	endif()

    add_subdirectory(third_party/googletest)
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
	add_subdirectory(test)
endif()
