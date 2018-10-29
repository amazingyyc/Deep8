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
#include "InputParameter.h"

namespace py = pybind11;

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
            .def("equalExceptBatch", &Shape::equalExceptBatch)
            .def("batchSize", &Shape::batchSize)
            .def("size", &Shape::size)
            .def("dim", &Shape::dim)
            .def("nDims", &Shape::nDims)
            .def("batch", &Shape::batch)
            .def("row", &Shape::row)
            .def("col", &Shape::col)
            .def("reShape", (void (Shape::*)(Shape &)) &Shape::reShape)
            .def("reShape", (void (Shape::*)(std::vector<size_t>)) &Shape::reShape)
            .def("reShape", (void (Shape::*)(size_t, Shape &)) &Shape::reShape);
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
            .def(py::init<T, bool, T>(), py::arg("lr") = 0.1, py::arg("cg") = false, py::arg("ct") = 5.0);

    /**AdagradTrainer*/
    className = std::string("AdagradTrainer") + suffix;
    py::class_<AdagradTrainer<T>, Trainer<T>>(m, className.c_str())
            .def(py::init<T, T, bool, T>(),
                    py::arg("learningRate") = 0.1, py::arg("epsilon") = 1e-7, py::arg("clipGradient") = false, py::arg("clipThreshold") = 5.0);

    /**AdamTrainer*/
    className = std::string("AdamTrainer") + suffix;
    py::class_<AdamTrainer<T>, Trainer<T>>(m, className.c_str())
            .def(py::init<T, T, T, T, bool, T>(),
                    py::arg("learningRate") = 0.1,
                    py::arg("beta1") = 0.9,
                    py::arg("beta2") = 0.999,
                    py::arg("epsilon") = 1e-7,
                    py::arg("clipGradient") = false,
                    py::arg("clipThreshold") = 5.0);

    /**RMSPropTrainer*/
    className = std::string("RMSPropTrainer") + suffix;
    py::class_<RMSPropTrainer<T>, Trainer<T>>(m, className.c_str())
        .def(py::init<T, T, T, bool, T>(),
                py::arg("learningRate") = 0.1,
                py::arg("decay") = 0.9,
                py::arg("epsilon") = 1e-7,
                py::arg("clipGradient") = false,
                py::arg("clipThreshold") = 5.0);

    /**MomentumTrainer*/
    className = std::string("MomentumTrainer") + suffix;
    py::class_<MomentumTrainer<T>, Trainer<T>>(m, className.c_str())
            .def(py::init<T, T, bool, T>(),
                    py::arg("learningRate") = 0.1,
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
            .def_readonly("value", &Class::value)
            .def_readonly("gradient", &Class::gradient);
}

/**
 * InputParameter
 */
template <typename T>
void declareInputParameter(py::module &m, const std::string &suffix = "") {
     using Class = InputParameter<T>;
     std::string className = std::string("InputParameter") + suffix;

     py::class_<Class, Parameter<T>>(m, className.c_str())
             .def_readwrite("value", &Class::value)
             .def("feed", [](InputParameter<T> &input, py::buffer buffer) {
                /**Request a buffer descriptor from Python*/
                 py::buffer_info info = buffer.request();

                 if (info.format != py::format_descriptor<T>::format()) {
                     DEEP8_RUNTIME_ERROR("The input format is not correct");
                 }

                 if (info.size < input.value.size()) {
                     DEEP8_RUNTIME_ERROR("The input size must be > " << input.value.size());
                 }

                 input.feed(info.ptr);
             });
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
            .def("tanh", &Class::tanh)
            .def("backward", &Class::backward)
            .def("valueStr", &Class::valueString)
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
            .def(T() / py::self);
}

/**
 * declare Expression Function
 */
template <typename T>
void declareExpressionFunction(py::module &m, const std::string &suffix = "") {
    m.def(("parameter" + suffix).c_str(), (Expression<T> (*)(Executor<T>*, std::vector<size_t>)) &parameter<T>);
    m.def(("parameter" + suffix).c_str(), (Expression<T> (*)(Executor<T>*, size_t, std::vector<size_t>)) &parameter<T>);
    m.def(("parameter" + suffix).c_str(), (Expression<T> (*)(Executor<T>*, Shape&)) &parameter<T>);

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

        return inputParameter<T>(executor, list, info.ptr);
    });

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

        return inputParameter<T>(executor, batch, list, info.ptr);
    });

    m.def(("inputParameter" + suffix).c_str(), [](Executor<T> *executor, Shape &shape, py::buffer buffer) -> Expression<T> {
        /**Request a buffer descriptor from Python*/
        py::buffer_info info = buffer.request();

        if (info.format != py::format_descriptor<T>::format()) {
            DEEP8_RUNTIME_ERROR("The input format is not correct");
        }

        auto size = shape.size();

        if (info.size < size) {
            DEEP8_RUNTIME_ERROR("The input size must be > " << size);
        }

        return inputParameter<T>(executor, shape, info.ptr);
    });

    m.def(("add" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, const Expression<T>&)) &add<T>);
    m.def(("add" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, T)) &add<T>);
    m.def(("add" + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &add<T>);

    m.def(("minus" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, const Expression<T>&)) &minus<T>);
    m.def(("minus" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, T)) &minus<T>);
    m.def(("minus" + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &minus<T>);

    m.def(("multiply" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, const Expression<T>&)) &multiply<T>);
    m.def(("multiply" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, T)) &multiply<T>);
    m.def(("multiply" + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &multiply<T>);

    m.def(("divide" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, const Expression<T>&)) &divide<T>);
    m.def(("divide" + suffix).c_str(), (Expression<T> (*)(const Expression<T>&, T)) &divide<T>);
    m.def(("divide" + suffix).c_str(), (Expression<T> (*)(T, const Expression<T>&)) &divide<T>);

    m.def(("abs" + suffix).c_str(), &abs<T>);
    m.def(("avgPooling2d" + suffix).c_str(), &avgPooling2d<T>);
    m.def(("conv2d" + suffix).c_str(), &conv2d<T>);
    m.def(("deConv2d" + suffix).c_str(), &deConv2d<T>);
    m.def(("exp" + suffix).c_str(), &exp<T>);
    m.def(("l1Norm" + suffix).c_str(), &l1Norm<T>);
    m.def(("l2Norm" + suffix).c_str(), &l2Norm<T>);
    m.def(("linear" + suffix).c_str(), &linear<T>);
    m.def(("log" + suffix).c_str(), &log<T>);
    m.def(("lReLu" + suffix).c_str(), &lReLu<T>);
    m.def(("matrixMultiply" + suffix).c_str(), &matrixMultiply<T>);
    m.def(("maxPooling2d" + suffix).c_str(), &maxPooling2d<T>);
    m.def(("reLu" + suffix).c_str(), &reLu<T>);

    m.def(("reShape" + suffix).c_str(), (Expression<T> (*)(const Expression<T> &, Shape &)) &reShape<T>);
    m.def(("reShape" + suffix).c_str(), (Expression<T> (*)(const Expression<T> &, std::vector<size_t>)) &reShape<T>);

    m.def(("sigmoid" + suffix).c_str(), &sigmoid<T>);
    m.def(("softmax" + suffix).c_str(), &softmax<T>);
    m.def(("square" + suffix).c_str(), &square<T>);
    m.def(("tanh" + suffix).c_str(), &tanh<T>);
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
            .def(py::init<Trainer<T>*, DeviceType, bool>(),
                 py::arg("tr") = nullptr,
                 py::arg("deviceType") = DeviceType::CPU,
                 py::arg("flag") = true)
            .def("addParameter", (Parameter<T>* (Class::*)(std::vector<size_t>)) &Class::addParameter)
            .def("addParameter", (Parameter<T>* (Class::*)(size_t, std::vector<size_t>)) &Class::addParameter)
            .def("addParameter", (Parameter<T>* (Class::*)(Shape&)) &Class::addParameter)
            .def("addInputParameter", [](Class &executor, std::vector<size_t> list, py::buffer buffer) -> InputParameter<T>* {
                /**Request a buffer descriptor from Python*/
                py::buffer_info info = buffer.request();

                if (info.format != py::format_descriptor<T>::format()) {
                    DEEP8_RUNTIME_ERROR("The input format is not correct");
                }

                auto size = Shape(1, list).size();

                if (info.size < size) {
                    DEEP8_RUNTIME_ERROR("The input size must be > " << size);
                }

                return executor.addInputParameter(list, info.ptr);
            })
            .def("addInputParameter", [](Class &executor, size_t batch, std::vector<size_t> list, py::buffer buffer) -> InputParameter<T>* {
                /**Request a buffer descriptor from Python*/
                py::buffer_info info = buffer.request();

                if (info.format != py::format_descriptor<T>::format()) {
                    DEEP8_RUNTIME_ERROR("The input format is not correct");
                }

                auto size = Shape(batch, list).size();

                if (info.size < size) {
                    DEEP8_RUNTIME_ERROR("The input size must be > " << size);
                }

                return executor.addInputParameter(batch, list, info.ptr);
            })
            .def("addInputParameter", [](Class &executor, Shape &shape, py::buffer buffer) -> InputParameter<T>* {
                /**Request a buffer descriptor from Python*/
                py::buffer_info info = buffer.request();

                if (info.format != py::format_descriptor<T>::format()) {
                    DEEP8_RUNTIME_ERROR("The input format is not correct");
                }

                auto size = shape.size();

                if (info.size < size) {
                    DEEP8_RUNTIME_ERROR("The input size must be > " << size);
                }

                return executor.addInputParameter(shape, info.ptr);
            });
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

    /**
     * InputParameter
     */
    declareInputParameter<float>(m);
    declareInputParameter<double>(m, "D");
#ifdef HAVE_HALF
    declareInputParameter<half>(m, "H");
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