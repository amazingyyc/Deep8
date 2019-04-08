#include "math/MaxIndex2d.h"
#include "model/Net.h"

namespace Deep8 {

void backward(Variable &v, bool clearInterim) {
    v.executor->backward(&v, clearInterim);
}

void backward(Variable *v, bool clearInterim) {
    v->executor->backward(v, clearInterim);
}

Variable& parameter(Executor *executor, std::vector<size_t> list, bool updateGradient, DType type) {
    return *(executor->addVariable(list, type, updateGradient));
}

Variable& parameter(Executor *executor, size_t batch, std::vector<size_t> list, bool updateGradient, DType type) {
    return *(executor->addVariable(batch, list, type, updateGradient));
}

Variable& parameter(Executor *executor, Shape& shape, bool updateGradient, DType type) {
    return *(executor->addVariable(shape, type, updateGradient));
}

Variable& inputParameter(Executor *executor, std::vector<size_t> list, DType type) {
    return *(executor->addVariable(list, type, false, false));
}

Variable& inputParameter(Executor *executor, size_t batch, std::vector<size_t> list, DType type) {
    return *(executor->addVariable(batch, list, type, false, false));
}

Variable& inputParameter(Executor *executor, Shape& shape, DType type) {
    return *(executor->addVariable(shape, type, false, false));
}

Variable& feed(Variable &v, const void *ptr) {
    return v.feed(ptr);
}

Variable& fetch(Variable &v, void *ptr) {
    return v.fetch(ptr);
}

Variable& constant(Variable &v, float scalar) {
    return v.constant(scalar);
}

Variable& zero(Variable &v) {
    return v.zero();
}

Variable& one(Variable &v) {
    return v.one();
}

Variable& gaussian(Variable &v, float mean, float stddev) {
    return v.gaussian(mean, stddev);
}

Variable& positiveUnitball(Variable &v) {
    return v.positiveUnitball();
}

Variable& random(Variable &v, float lower, float upper) {
    return v.random(lower, upper);
}

Variable& uniform(Variable &v, float left, float right) {
    return v.uniform(left, right);
}

Variable& assign(Variable &x, Variable &v) {
    return x.assign(v);
}

Variable& operator + (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return *(x.executor->addFunction(new Add(inputs)));
}

Variable& operator - (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return *(x.executor->addFunction(new Minus(inputs)));
}

Variable& operator * (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return *(x.executor->addFunction(new MatrixMultiply(inputs)));
}

Variable& operator / (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return *(x.executor->addFunction(new Divide(inputs)));
}

Variable& operator += (Variable &x, Variable &y) {
    return x + y;
}

Variable& operator -= (Variable &x, Variable &y) {
    return x - y;
}

Variable& operator *= (Variable &x, Variable &y) {
    return x * y;
}

Variable& operator /= (Variable &x, Variable &y) {
    return x / y;
}

Variable& operator + (Variable &x, float c) {
    return addConstant(x, c);
}

Variable& operator - (Variable &x, float c) {
    return minusConstant(x, c);
}

Variable& operator * (Variable &x, float c) {
    return multiplyConstant(x, c);
}

Variable& operator / (Variable &x, float c) {
    return divideConstant(x, c);
}

Variable& add(Variable &x, Variable &y) {
    return x + y;
}

Variable& minus(Variable &x, Variable &y) {
    return x - y;
}

Variable& multiply(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };
    return *(x.executor->addFunction(new Multiply(inputs)));
}

Variable& divide(Variable &x, Variable &y) {
    return x / y;
}

Variable& addConstant(Variable &x, float c) {
    return linear(x, 1, c);
}

Variable& minusConstant(Variable &x, float c) {
    return linear(x, 1, -c);
}

Variable& multiplyConstant(Variable &x, float c) {
    return linear(x, c, 0);
}

Variable& divideConstant(Variable &x, float c) {
    return linear(x, 1.0 / c, 0);
}

Variable& abs(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Abs(inputs)));
}

Variable& avgPooling2d( Variable &x,
                        bool covered, 
                        int filterHeight, 
                        int filterWidth, 
                        int strideY, 
                        int strideX) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new AvgPooling2d(inputs, covered, filterHeight, filterWidth, strideY, strideX)));
}

Variable& conv2d(Variable &x,
                Variable &filter,
                bool covered,
                int strideY, 
                int strideX, 
                int dilationY, 
                int dilationX) {
    std::vector<Node*> inputs = { &x, &filter };

    return *(x.executor->addFunction(new Conv2d(inputs, covered, strideY, strideX, dilationY, dilationX)));
}

Variable& crossEntropy(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };
    
    return *(x.executor->addFunction(new CrossEntropy(inputs)));
}

Variable& deConv2d( Variable &x,
                    Variable &filter,
                    bool covered, 
                    int strideY, 
                    int strideX) {
    std::vector<Node*> inputs = { &x, &filter };

    return *(x.executor->addFunction(new DeConv2d(inputs, covered, strideY, strideX)));
}

Variable& dot(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };

    return *(x.executor->addFunction(new Dot(inputs)));
}

Variable& exp(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Exp(inputs)));
}

Variable& l1Distance(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };

    return *(x.executor->addFunction(new L1Distance(inputs)));
}

Variable& l1Norm(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new L1Norm(inputs)));
}

Variable& l2Distance(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };

    return *(x.executor->addFunction(new L2Distance(inputs)));
}

Variable& l2Norm(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new L2Norm(inputs)));
}

Variable& linear(Variable &x, float a, float b) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Linear(inputs, a, b)));
}

Variable& log(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Log(inputs)));
}

Variable& logSoftmax(Variable &x, int axis) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new LogSoftmax(inputs, axis)));
}

Variable& lRelu(Variable &x, float a) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new LReLu(inputs, a)));
}

Variable& matrixMultiply(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y};

    return *(x.executor->addFunction(new MatrixMultiply(inputs)));
}

Variable& maxPooling2d( Variable &x,
                        bool covered, 
                        int filterHeight, 
                        int filterWidth, 
                        int strideY, 
                        int strideX) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new MaxPooling2d(inputs, 
                                                covered, 
                                                filterHeight, 
                                                filterWidth, 
                                                strideY, 
                                                strideX)));
}

/**this is return the index of max, and no need to calculate gradient in backward, so this will not added to the compute graph*/
Variable& maxIndex2d(Variable &x,
                    bool covered, 
                    int filterHeight, 
                    int filterWidth, 
                    int strideY, 
                    int strideX) {
    auto inputShape = x.shape();

    auto batch        = (int)inputShape.batch;
    auto inputHeight  = (int)inputShape.dim(0);
    auto inputWidth   = (int)inputShape.dim(1);
    auto inputChannel = (int)inputShape.dim(2);

    int outputHeight;
    int outputWidth;

    if (!covered) {
        outputHeight = (inputHeight - filterHeight) / strideY + 1;
        outputWidth  = (inputWidth  - filterWidth) / strideX + 1;
    } else {
        outputHeight = (inputHeight - 1) / strideY + 1;
        outputWidth  = (inputWidth  - 1) / strideX + 1;
    }

    DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the shape is error");

    Shape indexShape((size_t)batch, {(size_t)outputHeight, (size_t)outputWidth, (size_t)inputChannel});

    Variable *index = x.executor->addVariable(indexShape, DType::Int32, false, false);

    Math::MaxIndex2d(x.value, index->value, covered, filterHeight, filterWidth, strideY, strideX);

    return *index;
}


Variable& maxUnPooling2d(Variable &x,
                        Variable& index,
                        bool covered, 
                        int filterHeight, 
                        int filterWidth, 
                        int strideY, 
                        int strideX) {
    std::vector<Node*> inputs = { &x, &index};

    return *(x.executor->addFunction(new MaxUnPooling2d(inputs, 
                                                    covered, 
                                                    filterHeight, 
                                                    filterWidth, 
                                                    strideY, 
                                                    strideX)));
}

Variable& pRelu(Variable &x, Variable &p) {
    std::vector<Node*> inputs = { &x, &p};

    return *(x.executor->addFunction(new PReLu(inputs)));
}

Variable& reduceMean(Variable &x, std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new ReduceMean(inputs, axis, keepDims)));
}

Variable& reduceSum(Variable &x, std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new ReduceSum(inputs, axis, keepDims)));
}

Variable& relu(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new ReLu(inputs)));
}

Variable& reShape(Variable &x, Shape &shape) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new ReShape(inputs, shape)));
}

Variable& reShape(Variable &x, std::vector<size_t> list) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new ReShape(inputs, list)));
}

Variable& sigmoid(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Sigmoid(inputs)));
}

Variable& softmax(Variable &x, int axis) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Softmax(inputs, axis)));
}

Variable& sqrt(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Sqrt(inputs)));
}

Variable& square(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Square(inputs)));
}

Variable& tanh(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return *(x.executor->addFunction(new Tanh(inputs)));
}

Variable& l1Loss(Variable &x, Variable &y) {
    return reduceMean(abs(x - y), {}, false);
}

Variable& l1NormLoss(Variable &x) {
    return reduceMean(l1Norm(x), {}, false);
}

Variable& l2Loss(Variable &x, Variable &y) {
    return reduceMean(square(x - y), {}, false);
}

Variable& l2NormLoss(Variable &x) {
    return reduceMean(l2Norm(x), {}, false);
}

Variable& softmaxCrossEntropyLoss(Variable &x, Variable &y, int axis) {
    auto xshape = x.shape();
    auto yshape = y.shape();

    DEEP8_ARGUMENT_CHECK(xshape == yshape, "the shape of SoftmaxCrossEntropyLoss must be same");
	DEEP8_ARGUMENT_CHECK(-1 <= axis && axis < (int)xshape.nDims, "the axis is error");

	Variable& pred = logSoftmax(x, axis);
	Variable& mult = multiply(linear(y, -1, 0), pred);

	int sumAxis = axis;

	if (-1 != sumAxis) {
		sumAxis += 1;
	}

	Variable& sum = reduceSum(mult, { sumAxis }, false);

	return reduceMean(sum, {}, false);
}

}