#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Shape.h"
#include "Device.h"
#include "Trainer.h"
#include "Executor.h"
#include "EagerExecutor.h"
#include "Expression.h"
#include "Tensor.h"
#include "Parameter.h"
#include "InputParameter.h"

namespace py = pybind11;

namespace Deep8 {
namespace Python {

/**
 * declare the SGDTrainer
 */
template <typename T>
void declareSGDTrainer(py::module &m, const std::string &suffix) {
    using Class = SGDTrainer<T>;
    std::string className = std::string("SGDTrainer") + suffix;
    py::class_<Class>(m, className.c_str())
            .def(py::init<T, bool, T>(), py::arg("lr") = 0.1, py::arg("cg") = false, py::arg("ct") = 5.0);
}

/**
 * declare Parameter
 */
template <typename T>
void declareParameter(py::module &m, const std::string &suffix) {
    using Class = Parameter<T>;
    std::string className = std::string("Parameter") + suffix;

    py::class_<Class>(m, className.c_str())
            .def(py::init<Tensor<T>&, Tensor<T>&>());
}

 /**
  * InputParameter
  */
 template <typename T>
void declareInputParameter(py::module &m, const std::string &suffix) {
     using Class = InputParameter<T>;
     std::string className = std::string("InputParameter") + suffix;

     py::class_<Class>(m, className.c_str())
             .def(py::init<Tensor<T>&>())
             .def("feed", &InputParameter<T>::feed);
}

/**
 * Expression
 */
template <typename T>
void declareExpression(py::module &m, const std::string &suffix) {
    using Class = Expression<T>;
    std::string className = std::string("Expression") + suffix;

    py::class_<Class>(m, className.c_str())
            .def(py::init())
            .def(py::init<Executor<T>*, Node*>())
            .def("abs", &Class::abs)
            .def("exp", &Class::exp)
            .def("l1Norm", &Class::l1Norm)
            .def("l2Norm", &Class::l2Norm)
            .def("linear", &Class::linear)
            .def("log", &Class::log)
            .def("lReLu", &Class::lReLu)
            .def("reLu", &Class::reLu)
            .def("reShape", (Expression<T> (Class::*)(Shape&)) &Class::reShape)
            .def("reShape", (Expression<T> (Class::*)(std::vector<size_t>)) &Class::reShape)
            .def("sigmoid", &Class::sigmoid)
            .def("softmax", &Class::softmax)
            .def("square", &Class::square)
            .def("sumElements", &Class::sumElements)
            .def("tanH", &Class::tanH);
}

/**
 * declare the Eager Executor
 */
template <typename T>
void declareEagerExecutor(py::module &m, const std::string &suffix) {
    using Class = EagerExecutor<T>;
    std::string className = std::string("EagerExecutor") + suffix;

    py::class_<Class>(m, className.c_str())
            .def(py::init<Trainer<T>*, DeviceType, bool>(),
                 py::arg("tr") = nullptr,
                 py::arg("deviceType") = DeviceType::CPU,
                 py::arg("flag") = true)
            .def("addParameter", (Parameter<T>* (Class::*)(std::vector<size_t>)) &Class::addParameter)
            .def("addParameter", (Parameter<T>* (Class::*)(Shape&)) &Class::addParameter)
            .def("addInputParameter",
                 (InputParameter<T>* (Class::*)(std::vector<size_t>, void*)) &Class::addInputParameter,
                 py::arg("list") = std::vector<size_t>(),
                 py::arg("ptr") = nullptr)
            .def("addInputParameter",
                 (InputParameter<T>* (Class::*)(Shape&, void*)) &Class::addInputParameter,
                 py::arg("shape") = Shape(),
                 py::arg("ptr") = nullptr)
            .def("backward", (void (Class::*)(Expression<T> &)) &Class::backward)
            .def("backward", (void (Class::*)(Node*)) &Class::backward);
}

PYBIND11_MODULE(deep8, m) {
    /**
     * Shape
     */
    py::class_<Shape>(m, "Shape")
            .def(py::init())
            .def(py::init<std::vector<size_t>>())
            .def("equalExceptBatch", &Shape::equalExceptBatch)
            .def("batchSize", &Shape::batchSize)
            .def("size", &Shape::size)
            .def("dim", &Shape::dim)
            .def("nDims", &Shape::nDims)
            .def("batch", &Shape::batch)
            .def("row", &Shape::row)
            .def("col", &Shape::col)
            .def("reShape", (void (Shape::*)(Shape &)) &Shape::reShape)
            .def("reShape", (void (Shape::*)(std::initializer_list<size_t>)) &Shape::reShape)
            .def("reShape", (void (Shape::*)(size_t, Shape &)) &Shape::reShape);

    /**
     * DeviceType
     */
    py::enum_<DeviceType>(m, "DeviceType")
            .value("CPU", DeviceType::CPU)
            .value("GPU", DeviceType::GPU)
            .export_values();

    /**
     * SGDTrainer
     */
    declareSGDTrainer<float>(m, "");
    declareSGDTrainer<double>(m, "D");
#ifdef HAVE_HALF
    declareSGDTrainer<half>(m, "H");
#endif

    /**
     * Parameter
     */
    declareParameter<float>(m, "");
    declareParameter<double>(m, "D");
#ifdef HAVE_HALF
    declareParameter<half>(m, "H");
#endif

    /**
     * InputParameter
     */
    declareInputParameter<float>(m, "");
    declareInputParameter<double>(m, "D");
#ifdef HAVE_HALF
    declareInputParameter<half>(m, "H");
#endif

    /**
     * Expression
     */
    declareExpression<float>(m, "");
    declareExpression<double>(m, "D");
#ifdef HAVE_HALF
    declareExpression<half>(m, "H");
#endif

    /**
     * EagerExecutor
     */
    declareEagerExecutor<float>(m, "");
    declareEagerExecutor<double>(m, "D");
#ifdef HAVE_HALF
    declareEagerExecutor<half>(m, "H");
#endif
 }

}
}