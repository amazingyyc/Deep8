#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "model/Shape.h"
#include "model/Device.h"
#include "model/Executor.h"
#include "model/EagerExecutor.h"
#include "model/Expression.h"
#include "model/Tensor.h"
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
 * Variable
 */
void declareVariable(py::module &m) {
    py::class_<Variable, Node>(m, "Variable")
            .def_readonly("updateGradient", &Variable::updateGradient)
            .def_readonly("value", &Variable::value)
            .def_readonly("gradient", &Variable::gradient);
}

/**
 * Expression
 */
void declareExpression(py::module &m) {
    py::class_<Expression>(m, "Expression")
            .def(py::init())
            .def(py::init<Executor*, Node*>())
            .def("forward",     &Expression::forward)
            .def("backward",    &Expression::backward)
            .def("valueStr",    &Expression::valueStr)
            .def("constant",    &Expression::constant,
                                py::arg("scalar") = 0)
            .def("zero", &Expression::zero)
            .def("one", &Expression::one)
            .def("gaussian", &Expression::gaussian, 
                                py::arg("mean") = 0.0,
                                py::arg("stddev") = 0.01)
            .def("positiveUnitball", &Expression::positiveUnitball)
            .def("random", &Expression::random, 
                                py::arg("lower") = 0.0, 
                                py::arg("upper") = 1.0)
            .def("uniform", &Expression::uniform, 
                                py::arg("left") = 0.0, 
                                py::arg("right") = 1.0)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / py::self)
            .def("add",         &Expression::add)
            .def("minus",       &Expression::minus)
            .def("multiply",    &Expression::multiply)
            .def("divide",      &Expression::divide)
            .def("dot",         &Expression::dot)
            .def("abs",         &Expression::abs)
            .def("avgPooling2d", &Expression::avgPooling2d,
                            py::arg("covered") = false, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1)
            .def("conv2d", &Expression::conv2d, 
                            py::arg("filter"),
                            py::arg("covered") = false, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1, 
                            py::arg("dilationY") = 1, 
                            py::arg("dilationX") = 1)
            .def("crossEntropy", &Expression::crossEntropy)
            .def("deConv2d", &Expression::deConv2d, 
                            py::arg("filter"),
                            py::arg("covered") = false, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1)
            .def("exp",             &Expression::exp)
            .def("l1Distance",      &Expression::l1Distance)
            .def("l1Norm",          &Expression::l1Norm)
            .def("l2Distance",      &Expression::l2Distance)
            .def("l2Norm",          &Expression::l2Norm)
            .def("linear",          &Expression::linear)
            .def("log",             &Expression::log)
            .def("logSoftmax",      &Expression::logSoftmax, 
                                py::arg("axis") = -1)
            .def("lRelu", &Expression::lRelu)
            .def("matrixMultiply",  &Expression::matrixMultiply)
            .def("maxPooling2d",    &Expression::maxPooling2d, 
                            py::arg("covered") = false, 
                            py::arg("filterHeight") = 1, 
                            py::arg("filterWidth") = 1, 
                            py::arg("strideY") = 1, 
                            py::arg("strideX") = 1)
            .def("mean",        &Expression::mean)
            .def("pRelu",       &Expression::pRelu)
            .def("reduceMean",  &Expression::reduceMean,
                            py::arg("axis") =  -1,
                            py::arg("keep") =  false)
            .def("reduceSum", &Expression::reduceSum,
                            py::arg("axis") =  -1,
                            py::arg("keep") =  false)
            .def("relu", &Expression::relu)
            .def("reShape", (Expression (Expression::*)(Shape&)) &Expression::reShape)
            .def("reShape", (Expression (Expression::*)(std::vector<size_t>)) &Expression::reShape)
            .def("sigmoid", &Expression::sigmoid)
            .def("softmax", &Expression::softmax, py::arg("axis") = -1)
            .def("square",  &Expression::square)
            .def("sum",     &Expression::sum)
            .def("tanh",    &Expression::tanh)
            .def("feed", [](Expression* express, py::buffer buffer) {
                /**Request a buffer descriptor from Python*/
                py::buffer_info info = buffer.request();

                express->feed(info.ptr);
            })
            .def("fetch", [](Expression* express, py::buffer buffer) {
                /**Request a buffer descriptor from Python*/
                py::buffer_info info = buffer.request();

                express->fetch(info.ptr);
            })
            .def("l1DistanceLoss",  &Expression::l1DistanceLoss)
            .def("l1NormLoss",      &Expression::l1NormLoss)
            .def("l2DistanceLoss",  &Expression::l2DistanceLoss)
            .def("l2NormLoss",      &Expression::l2NormLoss)
            .def("softmaxCrossEntropyLoss", &Expression::softmaxCrossEntropyLoss);
}

/**
 * declare Expression Function
 */
void declareExpressionFunction(py::module &m) {
    /**parameter*/
    m.def("parameter", (Expression (*)(Executor*, std::vector<size_t>, bool, DType)) &parameter,
                        py::arg("executor"),
                        py::arg("list"),
                        py::arg("updateGradient") = true, 
                        py::arg("type") = DType::Float32);

    m.def("parameter", (Expression (*)(Executor*, size_t, std::vector<size_t>, bool, DType)) &parameter, 
                    py::arg("executor"),
                    py::arg("batch"),
                    py::arg("list"),
                    py::arg("updateGradient") = true, 
                    py::arg("type") = DType::Float32);

    m.def("parameter", (Expression (*)(Executor*, Shape&, bool, DType)) &parameter, 
                    py::arg("executor"),
                    py::arg("shape"),
                    py::arg("updateGradient") = true, 
                    py::arg("type") = DType::Float32);
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
        //.def("trainableParameters", &Executor::trainableParameters, pybind11::return_value_policy::reference);

    py::class_<EagerExecutor, Executor>(m, "EagerExecutor")
        .def(py::init<DeviceType, bool>(), 
            py::arg("deviceType") = DeviceType::CPU, 
            py::arg("flag") = true)
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
    declareExpression(m);
    declareExpressionFunction(m);
    declareLearningRateIterator(m);

    declareTrainer(m);
    declareExecutor(m);
 }

}
}