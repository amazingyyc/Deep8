project(deep8_test LANGUAGES CXX)

set(DEEP8_TEST_SRC_FILES

	Deep8Test.cpp
	)

message(STATUS "========================================================================================================")
message(STATUS "begin compile deep8_test")
message(STATUS "========================================================================================================")

add_definitions(-DEIGEN_USE_THREADS -DEIGEN_FAST_MATH -DEIGEN_NO_CUDA)

if (HAVE_CUDA)
	add_definitions(-DHAVE_CUDA)
endif()

add_executable(deep8_test ${DEEP8_TEST_SRC_FILES})
target_link_libraries(deep8_test deep8 gtest)

message(STATUS "========================================================================================================")
message(STATUS "end compile deep8_test")
message(STATUS "========================================================================================================")
