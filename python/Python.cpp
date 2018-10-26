#include <pybind11/pybind11.h>

#include "Shape.h"

namespace py = pybind11;

namespace Deep8 {
namespace Python {

PYBIND11_MODULE(deep8, m) {
    py::class_<Shape>(m, "Shape")
            .def(py::init())
            .def(py::init<std::initializer_list<size_t>>())
            .def(py::init<std::vector<size_t>>())
            .def(py::init<size_t, std::initializer_list<size_t>>())
            .def(py::init<const Shape&>())
            .def("equalExceptBatch", &Shape::equalExceptBatch)
            .def("batchSize", &Shape::batchSize)
            .def("size", &Shape::size)
            .def("dim", &Shape::dim)
            .def("nDims", &Shape::nDims)
            .def("batch", &Shape::batch)
            .def("row", &Shape::row)
            .def("col", &Shape::col)
            .def("reShape", (void (Shape::*)(Shape&)) &Shape::reShape)
            .def("reShape", (void (Shape::*)(std::initializer_list<size_t>)) &Shape::reShape)
            .def("reShape", (void (Shape::*)(size_t, Shape &)) &Shape::reShape);
}

}
}