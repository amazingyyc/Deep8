project(deep8)

pybind11_add_module(deep8 SHARED Python.cpp)
target_link_libraries(deep8 PRIVATE deep8_native)
