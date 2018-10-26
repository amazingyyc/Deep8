project(deep8_python)

pybind11_add_module(deep8_python Python.cpp)
target_link_libraries(deep8_python PRIVATE deep8)
