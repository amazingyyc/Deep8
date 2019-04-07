#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "nodes/Node.h"
#include "nodes/Variable.h"
#include "model/Shape.h"
#include "model/Device.h"
#include "model/Executor.h"
#include "model/EagerExecutor.h"
#include "model/Tensor.h"
#include "model/Net.h"
#include "trainer/ConstantLearningRateIterator.h"
#include "trainer/LinearDecayLearningRateIterator.h"
#include "trainer/Trainer.h"
#include "trainer/SGDTrainer.h"
#include "trainer/AdamTrainer.h"
#include "trainer/AdagradTrainer.h"
#include "trainer/MomentumTrainer.h"
#include "trainer/RMSPropTrainer.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace Deep8 {
namespace Python {

/**
 * DeviceType
 */
void declareDeviceType(py::module &m) {
    py::enum_<DeviceType>(m, "DeviceType")
                        .value("CPU", DeviceType::CPU)
                        .value("GPU", DeviceType::GPU)
                        .export_values();
}

/**
 * DType
 */
void declareDType(py::module &m) {
    py::enum_<DType>(m, "DType")
                        .value("UnKnown",   DType::UnKnown)
                        .value("Bool",      DType::Bool)
                        .value("Uint8",     DType::Uint8)
                        .value("Int8",      DType::Int8)
                        .value("Uint16",    DType::Uint16)
                        .value("Int16",     DType::Int16)
                        .value("Uint32",    DType::Uint32)
                        .value("Int32",     DType::Int32)
                        .value("Uint64",    DType::Uint64)
                        .value("Int64",     DType::Int64)
                        .value("Float16",   DType::Float16)
                        .value("Float32",   DType::Float32)
                        .value("Float64",   DType::Float64)
                        .export_values();
}

/**
 * ElementType
 */
void declareElementType(py::module &m) {
    py::class_<ElementType>(m, "ElementType")
            .def_readwrite("id", &ElementType::id)
            .def_readwrite("byteWidth", &ElementType::byteWidth)
            .def_readwrite("name", &ElementType::name)
            .def_static("from", (ElementType (*)(DType)) &ElementType::from);
}

/**
 * Shape
 */
void declareShape(py::module &m) {
    py::class_<Shape>(m, "Shape")
            .def(py::init())
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
            .def("toStr", &Shape::toStr);
}

/**
 * declare Tensor
 */
void declareTensor(py::module &m) {
    py::class_<Tensor>(m, "Tensor")
            .def_readonly("shape", &Tensor::shape)
            .def_readonly("elementType", &Tensor::elementType)
            .def("valueStr", &Tensor::valueStr);
}

/**
 * Node
 */
void declareNode(py::module &m) {
    py::class_<Node>(m, "Node");
}

/**
 * Variable
 */
void declareVariable(py::module &m) {
    py::class_<Variable, Node>(m, "Variable")
            .def_readonly("updateGradient", &Variable::updateGradient)
            .def_readonly("value",      &Variable::value)
            .def_readonly("gradient",   &Variable::gradient)
            .def("shape",           &Variable::shape)
            .def("elementType",     &Variable::elementType)
            .def("deviceType",      &Variable::deviceType)
            .def("isScalar",        &Variable::isScalar)
            .def("isScalar",        &Variable::isScalar)
            .def("zeroGradient",    &Variable::zeroGradient)
            .def("oneGradient",     &Variable::oneGradient)
            .def("removeGradient",  &Variable::removeGradient)
            .def("valueStr",        &Variable::valueStr)
            .def("constant",    &Variable::constant, py::arg("scalar") = 0, pybind11::return_value_policy::reference)
            .def("zero",        &Variable::zero, pybind11::return_value_policy::reference)
            .def("one",         &Variable::one, pybind11::return_value_policy::reference)
            .def("gaussian",    &Variable::gaussian, 
                                py::arg("mean") = 0.0, 
                                py::arg("stddev") = 0.01, 
                                pybind11::return_value_policy::reference)
            .def("positiveUnitball",    &Variable::positiveUnitball, pybind11::return_value_policy::reference)
            .def("random",    &Variable::random, 
                            py::arg("lower") = 0.0, 
                            py::arg("upper") = 1.0, 
                            pybind11::return_value_policy::reference)
            .def("uniform",   &Variable::uniform, 
                            py::arg("left") = 0.0, 
                            py::arg("right") = 1.0, 
                            pybind11::return_value_policy::reference)
            .def("assign", &Variable::assign, pybind11::return_value_policy::reference)
            .def("feed", [](Variable &v, py::buffer buffer) -> Variable& {
                py::buffer_info info = buffer.request();

                return v.feed(info.ptr);                 
            }, pybind11::return_value_policy::reference)
            .def("fetch", [](Variable &v, py::buffer buffer) -> Variable& {
                py::buffer_info info = buffer.request();

                return v.fetch(info.ptr);
            }, pybind11::return_value_policy::reference)
            .def("__add__", [](Variable &x, Variable &y) -> Variable& {
                return x - y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__sub__", [](Variable &x, Variable &y) -> Variable& {
                return x - y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__mul__", [](Variable &x, Variable &y) -> Variable& {
                return x * y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__div__", [](Variable &x, Variable &y) -> Variable& {
                return x / y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__add__", [](Variable &x, float c) -> Variable& {
                return x + c;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__sub__", [](Variable &x, float c) -> Variable& {
                return x - c;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__mul__", [](Variable &x, float c) -> Variable& {
                return x * c;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__div__", [](Variable &x, float c) -> Variable& {
                return x / c;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__iaddr__", [](Variable &x, Variable &y) -> Variable& {
                return x + y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__isub__", [](Variable &x, Variable &y) -> Variable& {
                return x - y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__imul__", [](Variable &x, Variable &y) -> Variable& {
                return x * y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("__idiv__", [](Variable &x, Variable &y) -> Variable& {
                return x / y;
            }, py::is_operator(), pybind11::return_value_policy::reference)
            .def("add",      &Variable::add,      pybind11::return_value_policy::reference)
            .def("minus",    &Variable::minus,    pybind11::return_value_policy::reference)
            .def("multiply", &Variable::multiply, pybind11::return_value_policy::reference)
            .def("divide",   &Variable::divide,   pybind11::return_value_policy::reference)
            .def("addConstant",         &Variable::addConstant, pybind11::return_value_policy::reference)
            .def("minusConstant",       &Variable::minusConstant, pybind11::return_value_policy::reference)
            .def("multiplyConstant",    &Variable::multiplyConstant, pybind11::return_value_policy::reference)
            .def("divideConstant",      &Variable::divideConstant, pybind11::return_value_policy::reference)
            .def("abs",      &Variable::abs, pybind11::return_value_policy::reference)
            .def("avgPooling2d", &Variable::avgPooling2d,
                            py::arg("covered") = true, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideY") = 1,
                            pybind11::return_value_policy::reference)
            .def("conv2d", &Variable::conv2d,
                            py::arg("filter"),
                            py::arg("covered") = true, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1, 
                            py::arg("dilationY") = 1, 
                            py::arg("dilationX") = 1,
                            pybind11::return_value_policy::reference)
            .def("crossEntropy", &Variable::crossEntropy, pybind11::return_value_policy::reference)
            .def("deConv2d", &Variable::deConv2d,
                            py::arg("filter"),
                            py::arg("covered") = true, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1,
                            pybind11::return_value_policy::reference)
            .def("dot",         &Variable::dot,         pybind11::return_value_policy::reference)
            .def("exp",         &Variable::exp,         pybind11::return_value_policy::reference)
            .def("l1Distance",  &Variable::l1Distance,  pybind11::return_value_policy::reference)
            .def("l1Norm",      &Variable::l1Norm,      pybind11::return_value_policy::reference)
            .def("l2Distance",  &Variable::l2Distance,  pybind11::return_value_policy::reference)
            .def("l2Norm",      &Variable::l2Norm,      pybind11::return_value_policy::reference)
            .def("linear",      &Variable::linear,      pybind11::return_value_policy::reference)
            .def("log",         &Variable::log,         pybind11::return_value_policy::reference)
            .def("logSoftmax",  &Variable::logSoftmax, py::arg("axis") = -1, pybind11::return_value_policy::reference)
            .def("lRelu",       &Variable::lRelu,       pybind11::return_value_policy::reference)
            .def("matrixMultiply", &Variable::matrixMultiply, pybind11::return_value_policy::reference)
            .def("maxPooling2d", &Variable::maxPooling2d,
                            py::arg("covered") = true, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1,
                            pybind11::return_value_policy::reference)
            .def("maxPooling2dWithIndex", &Variable::maxPooling2dWithIndex,
                            py::arg("index"), 
                            py::arg("covered") = true, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1,
                            pybind11::return_value_policy::reference)
            .def("maxUnPooling2d", &Variable::maxUnPooling2d,
                            py::arg("index"), 
                            py::arg("covered") = true, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1,
                            pybind11::return_value_policy::reference)
            .def("pRelu", &Variable::pRelu, pybind11::return_value_policy::reference)
            .def("reduceMean", &Variable::reduceMean,
                                    py::arg("axis") = std::vector<int>({-1}),  
                                    py::arg("keepDims") = true, 
                                    pybind11::return_value_policy::reference)
            .def("reduceSum",  &Variable::reduceSum, 
                                    py::arg("axis") = std::vector<int>({-1}),  
                                    py::arg("keepDims") = true,
                                    pybind11::return_value_policy::reference)
            .def("relu", &Variable::relu, pybind11::return_value_policy::reference)
            .def("reShape", (Variable& (Variable::*)(Shape&)) &Variable::reShape, pybind11::return_value_policy::reference)
            .def("reShape", (Variable& (Variable::*)(std::vector<size_t>)) &Variable::reShape, pybind11::return_value_policy::reference)
            .def("sigmoid", &Variable::sigmoid, pybind11::return_value_policy::reference)
            .def("softmax", &Variable::softmax, py::arg("axis") = -1, pybind11::return_value_policy::reference)
            .def("sqrt",            &Variable::sqrt,        pybind11::return_value_policy::reference)
            .def("square",          &Variable::square,      pybind11::return_value_policy::reference)
            .def("tanh",            &Variable::tanh,        pybind11::return_value_policy::reference)
            .def("l1Loss",          &Variable::l1Loss,      pybind11::return_value_policy::reference)
            .def("l1NormLoss",      &Variable::l1NormLoss,  pybind11::return_value_policy::reference)
            .def("l2Loss",          &Variable::l2Loss,      pybind11::return_value_policy::reference)
            .def("l2NormLoss",      &Variable::l2NormLoss,  pybind11::return_value_policy::reference)
            .def("softmaxCrossEntropyLoss", &Variable::softmaxCrossEntropyLoss, pybind11::return_value_policy::reference);
}

/**Net*/
void declareNet(py::module &m) {
    m.def("backward", (void (*)(Variable&, bool)) &backward,
                        py::arg("variable"),
                        py::arg("clearInterim") = true);

    m.def("parameter", (Variable& (*)(Executor*, std::vector<size_t>, bool, DType)) &parameter,
                        py::arg("executor"),
                        py::arg("list"),
                        py::arg("updateGradient") = true,
                        py::arg("type") = DType::Float32,
                        pybind11::return_value_policy::reference);
    
    m.def("parameter", (Variable& (*)(Executor*, size_t, std::vector<size_t>, bool, DType)) &parameter,
                    py::arg("executor"),
                    py::arg("batch"),
                    py::arg("list"),
                    py::arg("updateGradient") = true,
                    py::arg("type") = DType::Float32,
                    pybind11::return_value_policy::reference);

    m.def("parameter", (Variable& (*)(Executor*, Shape&, bool, DType)) &parameter,
                        py::arg("executor"),
                        py::arg("shape"),
                        py::arg("updateGradient") = true,
                        py::arg("type") = DType::Float32,
                        pybind11::return_value_policy::reference);

    m.def("inputParameter", (Variable& (*)(Executor*, std::vector<size_t>, DType)) &inputParameter,
                        py::arg("executor"),
                        py::arg("list"),
                        py::arg("type") = DType::Float32,
                        pybind11::return_value_policy::reference);

    m.def("inputParameter", (Variable& (*)(Executor*, size_t, std::vector<size_t>, DType)) &inputParameter,
                py::arg("executor"),
                py::arg("batch"),
                py::arg("list"),
                py::arg("type") = DType::Float32,
                pybind11::return_value_policy::reference);

    m.def("inputParameter", (Variable& (*)(Executor*, Shape&, DType)) &inputParameter,
                        py::arg("executor"),
                        py::arg("shape"),
                        py::arg("type") = DType::Float32,
                        pybind11::return_value_policy::reference);

    m.def("add", &add, pybind11::return_value_policy::reference);
    m.def("minus", &minus, pybind11::return_value_policy::reference);
    m.def("multiply", &multiply, pybind11::return_value_policy::reference);
    m.def("divide", &divide, pybind11::return_value_policy::reference);

    m.def("addConstant", &addConstant, pybind11::return_value_policy::reference);
    m.def("minusConstant", &minusConstant, pybind11::return_value_policy::reference);
    m.def("multiplyConstant", &multiplyConstant, pybind11::return_value_policy::reference);
    m.def("divideConstant", &divideConstant, pybind11::return_value_policy::reference);
    
    m.def("abs", &abs, pybind11::return_value_policy::reference);

    m.def("avgPooling2d", &avgPooling2d,
                py::arg("variable"),
                py::arg("covered") = true, 
                py::arg("filterHeight") = 1, 
                py::arg("filterWidth") = 1, 
                py::arg("strideY") = 1, 
                py::arg("strideX") = 1,
                pybind11::return_value_policy::reference);
    
    m.def("conv2d", &conv2d,
                py::arg("variable"),
                py::arg("filter"),
                py::arg("covered") = true, 
                py::arg("strideY") = 1, 
                py::arg("strideX") = 1, 
                py::arg("dilationY") = 1, 
                py::arg("dilationX") = 1,
                pybind11::return_value_policy::reference);

    m.def("crossEntropy", &crossEntropy, pybind11::return_value_policy::reference);

    m.def("deConv2d", &deConv2d,
            py::arg("variable"),
            py::arg("filter"),
            py::arg("covered") = true, 
            py::arg("strideY") = 1, 
            py::arg("strideX") = 1, 
            pybind11::return_value_policy::reference);

    m.def("dot", &dot, pybind11::return_value_policy::reference);
    m.def("exp", &exp, pybind11::return_value_policy::reference);
    m.def("l1Distance", &l1Distance, pybind11::return_value_policy::reference);
    m.def("l1Norm", &l1Norm, pybind11::return_value_policy::reference);
    m.def("l2Distance", &l2Distance, pybind11::return_value_policy::reference);
    m.def("l2Norm", &l2Norm, pybind11::return_value_policy::reference);

    m.def("linear", &linear,
            py::arg("variable"), 
            py::arg("a") = 1.0, 
            py::arg("b") = 0.0,
            pybind11::return_value_policy::reference);

    m.def("log", &log, pybind11::return_value_policy::reference);

    m.def("logSoftmax", &logSoftmax,
                py::arg("variable"), 
                py::arg("axis") = -1, 
                pybind11::return_value_policy::reference);

    m.def("lRelu", &lRelu,
            py::arg("variable"), 
            py::arg("a") = 0.0, 
            pybind11::return_value_policy::reference);

    m.def("matrixMultiply", &matrixMultiply, pybind11::return_value_policy::reference);

    m.def("maxPooling2d", &maxPooling2d,
                py::arg("variable"),
                py::arg("covered") = true, 
                py::arg("filterHeight") = 1, 
                py::arg("filterWidth") = 1, 
                py::arg("strideY") = 1, 
                py::arg("strideX") = 1,
                pybind11::return_value_policy::reference);

    m.def("maxPooling2dWithIndex", &maxPooling2dWithIndex,
            py::arg("variable"),
            py::arg("index"),
            py::arg("covered") = true, 
            py::arg("filterHeight") = 1, 
            py::arg("filterWidth") = 1, 
            py::arg("strideY") = 1, 
            py::arg("strideX") = 1,
            pybind11::return_value_policy::reference);

    m.def("maxUnPooling2d", &maxUnPooling2d,
            py::arg("variable"),
            py::arg("index"),
            py::arg("covered") = true, 
            py::arg("filterHeight") = 1, 
            py::arg("filterWidth") = 1, 
            py::arg("strideY") = 1, 
            py::arg("strideX") = 1,
            pybind11::return_value_policy::reference);

    m.def("pRelu", &pRelu, pybind11::return_value_policy::reference);

    m.def("reduceMean", &reduceMean,
                py::arg("variable"), 
                py::arg("axis") = std::vector<int>({-1}),  
                py::arg("keepDims") = true,
                pybind11::return_value_policy::reference);

    m.def("reduceSum", &reduceSum, 
                py::arg("variable"),
                py::arg("axis") = std::vector<int>({-1}),  
                py::arg("keepDims") = true,
                pybind11::return_value_policy::reference);

    m.def("relu", &relu, pybind11::return_value_policy::reference);

    m.def("reShape", (Variable& (*)(Variable&, Shape&)) &reShape, 
                py::arg("variable"),
                py::arg("shape"),
                pybind11::return_value_policy::reference);

    m.def("reShape", (Variable& (*)(Variable&, std::vector<size_t>)) &reShape, 
            py::arg("variable"),
            py::arg("shape"),
            pybind11::return_value_policy::reference);

    m.def("sigmoid", &sigmoid, pybind11::return_value_policy::reference);

    m.def("softmax", &softmax,
                    py::arg("variable"), 
                    py::arg("axis") = -1,  
                    pybind11::return_value_policy::reference);

    m.def("sqrt",   &sqrt,   pybind11::return_value_policy::reference);
    m.def("square", &square, pybind11::return_value_policy::reference);
    m.def("tanh",   &tanh,   pybind11::return_value_policy::reference);

    m.def("l1Loss", &l1Loss, pybind11::return_value_policy::reference);
    m.def("l1NormLoss", &l1NormLoss, pybind11::return_value_policy::reference);
    m.def("l2Loss", &l2Loss, pybind11::return_value_policy::reference);
    m.def("l2NormLoss", &l2NormLoss, pybind11::return_value_policy::reference);
    m.def("softmaxCrossEntropyLoss", &softmaxCrossEntropyLoss, pybind11::return_value_policy::reference);
}

/**
 * LearningRateIterator 
 */
void declareLearningRateIterator(py::module &m) {
    py::class_<LearningRateIterator>(m, "LearningRateIterator");

    py::class_<ConstantLearningRateIterator, LearningRateIterator>(m, "ConstantLearningRateIterator")
        .def(py::init<float>(), py::arg("lr") = 0.01);

    py::class_<LinearDecayLearningRateIterator, LearningRateIterator>(m, "LinearDecayLearningRateIterator")
        .def(py::init<int64_t, float, float>(),
                py::arg("totalStep"),
                py::arg("start") = 0.01,
                py::arg("end") = 0.0);
}

/**
 * declare the Trainer
 */
void declareTrainer(py::module &m) {
    /**Trainer*/
    py::class_<Trainer>(m, "Trainer")
        .def("train", (void (Trainer::*)(Executor*)) &Trainer::train);

    /**SGDTrainer*/
    py::class_<SGDTrainer, Trainer>(m, "SGDTrainer")
            .def(py::init<LearningRateIterator*, float>(), 
                py::arg("learningRate"), 
                py::arg("weightDecay") = 0);

    /**AdagradTrainer*/
    py::class_<AdagradTrainer, Trainer>(m, "AdagradTrainer")
            .def(py::init<LearningRateIterator*, float, float>(),
                    py::arg("learningRate"), 
                    py::arg("epsilon") = 1e-7, 
                    py::arg("weightDecay") = 0);

    /**AdamTrainer*/
    py::class_<AdamTrainer, Trainer>(m, "AdamTrainer")
            .def(py::init<LearningRateIterator*, float, float, float, float>(),
                    py::arg("learningRate"),
                    py::arg("beta1") = 0.9,
                    py::arg("beta2") = 0.999,
                    py::arg("epsilon") = 1e-7,
                    py::arg("weightDecay") = 0);

    /**RMSPropTrainer*/
    py::class_<RMSPropTrainer, Trainer>(m, "RMSPropTrainer")
        .def(py::init<LearningRateIterator*, float, float, float>(),
                py::arg("learningRate"),
                py::arg("rho") = 0.9,
                py::arg("epsilon") = 1e-7,
                py::arg("weightDecay") = 0);

    /**MomentumTrainer*/
    py::class_<MomentumTrainer, Trainer>(m, "MomentumTrainer")
            .def(py::init<LearningRateIterator*, float, float>(),
                    py::arg("learningRate"),
                    py::arg("alpha") = 0.9,
                    py::arg("weightDecay") = 0);
}

/**
 * Executor
 */
void declareExecutor(py::module &m) {
    py::class_<Executor>(m, "Executor");

    py::class_<EagerExecutor, Executor>(m, "EagerExecutor")
        .def(py::init<DeviceType>(), 
            py::arg("deviceType") = DeviceType::CPU)
        .def("clearInterimNodes", &EagerExecutor::clearInterimNodes);
}

PYBIND11_MODULE(deep8, m) {
    declareDeviceType(m);
    declareDType(m);
    declareElementType(m);
    declareNode(m);
    declareShape(m);

    declareTensor(m);
    declareVariable(m);
    declareLearningRateIterator(m);

    declareNet(m);

    declareTrainer(m);
    declareExecutor(m);
 }

}
}
