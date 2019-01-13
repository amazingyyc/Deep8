#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "Shape.h"
#include "Device.h"
#include "Trainer.h"
#include "Executor.h"
#include "EagerExecutor.h"
#include "Expression.h"
#include "Tensor.h"
#include "Parameter.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace Deep8 {
namespace Python {

/**
 * Device
 */
void declareDevice(py::module &m) {
    /**DeviceType*/
    py::enum_<DeviceType>(m, "deviceType")
            .value("CPU", DeviceType::CPU)
            .value("GPU", DeviceType::GPU)
            .export_values();
}

/**
 * Node
 */
void declareNode(py::module &m) {
    py::class_<Node>(m, "Node");
}

/**
 * Shape
 */
void declareShape(py::module &m) {
    py::class_<Shape>(m, "Shape")
            .def(py::init())
            .def(py::init<std::vector<size_t>&>())
            .def(py::init<std::vector<size_t>>())
            .def(py::init<size_t, std::vector<size_t>>())
            .def_readwrite("batch", &Shape::batch)
            .def_readwrite("nDims", &Shape::nDims)
            .def("reShape", (void (Shape::*)(Shape &)) &Shape::reShape)
            .def("equalExceptBatch", &Shape::equalExceptBatch)
            .def("size", &Shape::size)
            .def("batchSize", &Shape::batchSize)
            .def("dim", &Shape::dim)
            .def("stride", &Shape::stride)
            .def("row", &Shape::row)
            .def("col", &Shape::col)
            .def("toString", &Shape::toString);
}

/**
 * declare the Trainer
 */
template <typename T>
void declareTrainer(py::module &m, const std::string &suffix = "") {
    /**Trainer*/
    std::string className = std::string("Trainer") + suffix;
    py::class_<Trainer<T>>(m, className.c_str());

    /**SGDTrainer*/
    className = std::string("SGDTrainer") + suffix;
    py::class_<SGDTrainer<T>, Trainer<T>>(m, className.c_str())
            .def(py::init<T, bool, T>(), py::arg("lr") = 1e-3, py::arg("cg") = false, py::arg("ct") = 5.0);

    /**AdagradTrainer*/
    className = std::string("AdagradTrainer") + suffix;
    py::class_<AdagradTrainer<T>, Trainer<T>>(m, className.c_str())
            .def(py::init<T, T, bool, T>(),
                    py::arg("learningRate") = 1e-3, py::arg("epsilon") = 1e-7, py::arg("clipGradient") = false, py::arg("clipThreshold") = 5.0);

    /**AdamTrainer*/
    className = std::string("AdamTrainer") + suffix;
    py::class_<AdamTrainer<T>, Trainer<T>>(m, className.c_str())
            .def(py::init<T, T, T, T, bool, T>(),
                    py::arg("learningRate") = 1e-3,
                    py::arg("beta1") = 0.9,
                    py::arg("beta2") = 0.999,
                    py::arg("epsilon") = 1e-7,
                    py::arg("clipGradient") = false,
                    py::arg("clipThreshold") = 5.0);

    /**RMSPropTrainer*/
    className = std::string("RMSPropTrainer") + suffix;
    py::class_<RMSPropTrainer<T>, Trainer<T>>(m, className.c_str())
        .def(py::init<T, T, T, bool, T>(),
                py::arg("learningRate") = 1e-3,
                py::arg("decay") = 0.9,
                py::arg("epsilon") = 1e-7,
                py::arg("clipGradient") = false,
                py::arg("clipThreshold") = 5.0);

    /**MomentumTrainer*/
    className = std::string("MomentumTrainer") + suffix;
    py::class_<MomentumTrainer<T>, Trainer<T>>(m, className.c_str())
            .def(py::init<T, T, bool, T>(),
                    py::arg("learningRate") = 1e-3,
                    py::arg("alpha") = 0.9,
                    py::arg("clipGradient") = false,
                    py::arg("clipThreshold") = 5.0);
}

/**
 * declare Tensor
 */
template <typename T>
void declareTensor(py::module &m, const std::string &suffix = "") {
    using Class = Tensor<T>;
    std::string className = std::string("Tensor") + suffix;

    py::class_<Class>(m, className.c_str())
            .def_readwrite("shape", &Class::shape)
            .def("__str__", &Class::toString)
            .def("valueStr", &Class::valueString);
}

/**
 * Variable
 */
template <typename T>
void declareVariable(py::module &m, const std::string &suffix = "") {
    using Class = Variable<T>;
    std::string className = std::string("Variable") + suffix;

    py::class_<Class, Node>(m, className.c_str())
            .def_readonly("updateGradient", &Class::updateGradient)
            .def_readonly("value", &Class::value)
            .def_readonly("gradient", &Class::gradient);
}

/**
 * declare Parameter
 */
template <typename T>
void declareParameter(py::module &m, const std::string &suffix = "") {
    using Class = Parameter<T>;
    std::string className = std::string("Parameter") + suffix;

    py::class_<Class, Variable<T>>(m, className.c_str())
            .def_readonly("updateGradient", &Class::updateGradient)
            .def_readonly("value", &Class::value)
            .def_readonly("gradient", &Class::gradient);
}

/**
 * Expression
 */
template <typename T>
void declareExpression(py::module &m, const std::string &suffix = "") {
    using Class = Expression<T>;
    std::string className = std::string("Expression") + suffix;

    py::class_<Class>(m, className.c_str())
            .def(py::init())
            .def(py::init<Executor<T>*, Node*>())
            .def("forward", &Class::forward)
            .def("backward", &Class::backward)
            .def("valueString", &Class::valueString)
            .def("add", (Class (Class::*)(const Class&)) &Class::add)
            .def("add", (Class (Class::*)(T)) &Class::add)
            .def("minus", (Class (Class::*)(const Class&)) &Class::minus)
            .def("minus", (Class (Class::*)(T)) &Class::minus)
            .def("multiply", (Class (Class::*)(const Class&)) &Class::multiply)
            .def("multiply", (Class (Class::*)(T)) &Class::multiply)
            .def("divide", (Class (Class::*)(const Class&)) &Class::divide)
            .def("divide", (Class (Class::*)(T)) &Class::divide)
            .def("abs", &Class::abs)
            .def("avgPooling2d", &Class::avgPooling2d,
                            py::arg("covered") = false, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1)
            .def("conv2d", &Class::conv2d, 
                            py::arg("filter"),
                            py::arg("covered") = false, 
                            py::arg("strideH") = 1, 
                            py::arg("strideW") = 1, 
                            py::arg("dilationY") = 1, 
                            py::arg("dilationX") = 1)
            .def("crossEntropy", &Class::crossEntropy)
            .def("deConv2d", &Class::deConv2d, 
                            py::arg("filter"),
                            py::arg("covered") = false, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1)
            .def("exp", &Class::exp)
            .def("l1Norm", &Class::l1Norm)
            .def("l2Norm", &Class::l2Norm)
            .def("linear", &Class::linear)
            .def("log", &Class::log)
            .def("logSoftmax", &Class::logSoftmax, py::arg("axis") = -1)
            .def("lRelu", &Class::lRelu)
            .def("matrixMultiply", &Class::matrixMultiply)
            .def("maxPooling2d", &Class::maxPooling2d, 
                            py::arg("covered") = false, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1)
            .def("relu", &Class::relu)
            .def("reShape", (Expression<T> (Class::*)(Shape&)) &Class::reShape)
            .def("reShape", (Expression<T> (Class::*)(std::vector<size_t>)) &Class::reShape)
            .def("sigmoid", &Class::sigmoid)
            .def("softmax", &Class::softmax, py::arg("axis") = -1)
            .def("square", &Class::square)
            .def("tanh", &Class::tanh)
            .def(py::self + py::self)
            .def(py::self + T())
            .def(T() + py::self)
            .def(py::self - py::self)
            .def(py::self - T())
            .def(T() - py::self)
            .def(py::self * py::self)
            .def(py::self * T())
            .def(T() * py::self)
            .def(py::self / py::self)
            .def(py::self / T())
            .def(T() / py::self)
            .def("feed", [](Class* express, py::buffer buffer) {
                /**Request a buffer descriptor from Python*/
                py::buffer_info info = buffer.request();

                if (info.format != py::format_descriptor<T>::format()) {
                    DEEP8_RUNTIME_ERROR("The input format is not correct");
                }

                express->feed(info.ptr);
            })
            .def("fetch", [](Class* express, py::buffer buffer) {
                /**Request a buffer descriptor from Python*/
                py::buffer_info info = buffer.request();

                if (info.format != py::format_descriptor<T>::format()) {
                    DEEP8_RUNTIME_ERROR("The input format is not correct");
                }

                express->fetch(info.ptr);
            });
}

/**
 * declare Expression Function
 */
template <typename T>
void declareExpressionFunction(py::module &m, const std::string &suffix = "") {
    /**parameter*/
    m.def(("parameter" + suffix).c_str(), [](Executor<T> *executor, size_t batch, std::vector<size_t> list) -> Expression<T> {
        return parameter<T>(executor, batch, list, true, nullptr);
    });

    m.def(("parameter" + suffix).c_str(), [](Executor<T> *executor, std::vector<size_t> list) -> Expression<T> {
        return parameter<T>(executor, 1, list, true, nullptr);
    });

    /**input parameter*/
    m.def(("inputParameter" + suffix).c_str(), [](Executor<T> *executor, size_t batch, std::vector<size_t> list) -> Expression<T> {
        return parameter<T>(executor, batch, list, false, nullptr);
    });

    m.def(("inputParameter" + suffix).c_str(), [](Executor<T> *executor, std::vector<size_t> list) -> Expression<T> {
        return parameter<T>(executor, 1, list, false, nullptr);
    });

    /**input parameter with buffer*/
    m.def(("inputParameter" + suffix).c_str(), [](Executor<T> *executor, size_t batch, std::vector<size_t> list, py::buffer buffer) -> Expression<T> {
        /**Request a buffer descriptor from Python*/
        py::buffer_info info = buffer.request();

        if (info.format != py::format_descriptor<T>::format()) {
            DEEP8_RUNTIME_ERROR("The input format is not correct");
        }

        auto size = Shape(batch, list).size();

        if (info.size < size) {
            DEEP8_RUNTIME_ERROR("The input size must be > " << size);
        }

        return parameter<T>(executor, batch, list, false, info.ptr);
    });

    m.def(("inputParameter" + suffix).c_str(), [](Executor<T> *executor, std::vector<size_t> list, py::buffer buffer) -> Expression<T> {
        /**Request a buffer descriptor from Python*/
        py::buffer_info info = buffer.request();

        if (info.format != py::format_descriptor<T>::format()) {
            DEEP8_RUNTIME_ERROR("The input format is not correct");
        }

        auto size = Shape(1, list).size();

        if (info.size < size) {
            DEEP8_RUNTIME_ERROR("The input size must be > " << size);
        }

        return parameter<T>(executor, 1, list, false, info.ptr);
    });

    m.def(("add"      + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &add<T>);
    m.def(("minus"    + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &minus<T>);
    m.def(("multiply" + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &multiply<T>);
    m.def(("divide"   + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &divide<T>);
}

/**
 * Executor
 */
template <typename T>
void declareExecutor(py::module &m, const std::string &suffix = "") {
    using Class = Executor<T>;
    std::string className = std::string("Executor") + suffix;

    py::class_<Class>(m, className.c_str());
}

/**
 * declare the Eager Executor
 */
template <typename T>
void declareEagerExecutor(py::module &m, const std::string &suffix = "") {
    using Class = EagerExecutor<T>;
    std::string className = std::string("EagerExecutor") + suffix;

    py::class_<Class, Executor<T>>(m, className.c_str())
            .def(py::init<Trainer<T>*, DeviceType, bool>(), py::arg("tr") = nullptr, py::arg("deviceType") = DeviceType::CPU, py::arg("flag") = true)
            .def("clearIntermediaryNodes", &Class::clearIntermediaryNodes);
}

PYBIND11_MODULE(deep8, m) {
    declareDevice(m);
    declareNode(m);
    declareShape(m);

    declareTrainer<float>(m);
    declareTrainer<double>(m, "D");
#ifdef HAVE_HALF
    declareTrainer<half>(m, "H");
#endif

    declareTensor<float>(m);
    declareTensor<double>(m, "D");
#ifdef HAVE_HALF
    declareTensor<half>(m, "H");
#endif

    declareVariable<float>(m);
    declareVariable<double>(m, "D");
#ifdef HAVE_HALF
    declareVariable<half>(m, "H");
#endif

    declareParameter<float>(m);
    declareParameter<double>(m, "D");
#ifdef HAVE_HALF
    declareParameter<half>(m, "H");
#endif

    declareExpression<float>(m);
    declareExpression<double>(m, "D");
#ifdef HAVE_HALF
    declareExpression<half>(m, "H");
#endif

    declareExpressionFunction<float>(m);
    declareExpressionFunction<double>(m, "D");
#ifdef HAVE_HALF
    declareExpressionFunction<half>(m, "H");
#endif

    declareExecutor<float>(m);
    declareExecutor<double>(m, "D");
#ifdef HAVE_HALF
    declareExecutor<half>(m, "H");
#endif

    declareEagerExecutor<float>(m);
    declareEagerExecutor<double>(m, "D");
#ifdef HAVE_HALF
    declareEagerExecutor<half>(m, "H");
#endif
 }

}
}