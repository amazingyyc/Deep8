project(deep8_python LANGUAGES CXX)

set(DEEP8_PYTHON_SRC_FILES

	Python.cpp
	)

add_executable(deep8_python ${DEEP8_PYTHON_SRC_FILES})
target_link_libraries(deep8_python deep8)