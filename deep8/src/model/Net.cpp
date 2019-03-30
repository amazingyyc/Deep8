#include "model/Net.h"

namespace Deep8 {

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

Variable& dense(Variable &x, std::string weightName, int channel) {
    auto weight = x.executor->retainVariableByName(weightName);

    if (nullptr == weight) {
        auto xshape = x.shape();

        DEEP8_ARGUMENT_CHECK(1 == xshape.nDims || 2 == xshape.nDims, "the dense input nDims must be 1/2");

        Shape weightShape({size_t(channel), xshape.row()});
        
        auto weightVar = x.executor->addVariable(weightName, weightShape, x.value.elementType, true, true);

        weight = &weightVar;
    }

    std::vector<Node*> inputs = { weight, &x};

    return x.executor->addFunction(new MatrixMultiply(inputs));
}

Variable& pRelu(Variable &x, std::string pName) {
    auto p = x.executor->retainVariableByName(pName);

    if (nullptr == p) {
        auto pshape = x.shape();
        auto pvar   = x.executor->addVariable(pName, pshape, x.value.elementType, true, true);

        p = &pvar;
    }

    std::vector<Node*> inputs = { &x, p};

    return x.executor->addFunction(new PReLu(inputs));
}

Variable& conv2d(Variable &x,
                std::string filterName,
                int outputChannel,
                int filterHeight,
                int filterWidth,
                bool covered,
                int strideY, 
                int strideX, 
                int dilationY, 
                int dilationX) {
    /**if the filter have been create*/
    auto filter = x.executor->retainVariableByName(filterName);

    /**if not created, create it*/
    if (nullptr == filter) {
        auto xshape = x.shape();

        DEEP8_ARGUMENT_CHECK(3 == xshape.nDims, "the conv2d input nDims must be 3");
        DEEP8_ARGUMENT_CHECK(outputChannel >= 1 
                            && filterHeight >= 1
                            && filterWidth >= 1
                            && strideY >= 1
                            && strideX >= 1
                            && dilationY >= 1
                            && dilationX >= 1, "the parameter is error");
        
        auto inputChannel = xshape.dim(2);

        Shape filterShape({size_t(outputChannel), size_t(filterHeight), size_t(filterWidth), inputChannel});

        auto filerVar = x.executor->addVariable(filterName, filterShape, x.value.elementType, true, true);

        filter = &filerVar;
    }

    std::vector<Node*> inputs = { &x, filter };

    return x.executor->addFunction(new Conv2d(inputs, covered, strideY, strideX, dilationY, dilationX));
}

Variable& deConv2d( Variable &x,
                    std::string filterName,
                    int outputChannel,
                    int filterHeight,
                    int filterWidth,
                    bool covered, 
                    int strideY, 
                    int strideX) {
    /**if the filter have been create*/
    auto filter = x.executor->retainVariableByName(filterName);

    /**if not created, create it*/
    if (nullptr == filter) {
        auto xshape = x.shape();

        DEEP8_ARGUMENT_CHECK(3 == xshape.nDims, "the DeConv2d input nDims must be 3");
        DEEP8_ARGUMENT_CHECK(outputChannel >= 1 
                            && filterHeight >= 1
                            && filterWidth >= 1
                            && strideY >= 1
                            && strideX >= 1, "the parameter is error");
        
        auto inputChannel = xshape.dim(2);

        Shape filterShape({size_t(outputChannel), size_t(filterHeight), size_t(filterWidth), inputChannel});

        auto filerVar = x.executor->addVariable(filterName, filterShape, x.value.elementType, true, true);

        filter = &filerVar;
    }

    std::vector<Node*> inputs = { &x, filter };

    return x.executor->addFunction(new DeConv2d(inputs, covered, strideY, strideX));
}

Variable& operator + (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return x.executor->addFunction(new Add(inputs));
}

Variable& operator - (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return x.executor->addFunction(new Minus(inputs));
}

Variable& operator * (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return x.executor->addFunction(new MatrixMultiply(inputs));
}

Variable& operator / (Variable &x, Variable &y) {
    std::vector<Node*> inputs = {&x, &y};

    return x.executor->addFunction(new Divide(inputs));
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
    return x.executor->addFunction(new Multiply(inputs));
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

Variable& dot(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };

    return x.executor->addFunction(new Dot(inputs));
}

Variable& abs(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Abs(inputs));
}

Variable& avgPooling2d( Variable &x,
                        bool covered, 
                        int filterHeight, 
                        int filterWidth, 
                        int strideY, 
                        int strideX) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new AvgPooling2d(inputs, covered, filterHeight, filterWidth, strideY, strideX));
}

Variable& crossEntropy(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };
    
    return x.executor->addFunction(new CrossEntropy(inputs));
}

Variable& exp(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Exp(inputs));
}

Variable& l1Distance(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };

    return x.executor->addFunction(new L1Distance(inputs));
}

Variable& l1Norm(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new L1Norm(inputs));
}

Variable& l2Distance(Variable &x, Variable &y) {
    std::vector<Node*> inputs = { &x, &y };

    return x.executor->addFunction(new L2Distance(inputs));
}

Variable& l2Norm(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new L2Norm(inputs));
}

Variable& linear(Variable &x, float a, float b) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Linear(inputs, a, b));
}

Variable& log(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Log(inputs));
}

Variable& logSoftmax(Variable &x, int axis) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new LogSoftmax(inputs, axis));
}

Variable& lRelu(Variable &x, float a) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new LReLu(inputs, a));
}

Variable& maxPooling2d( Variable &x,
                        bool covered, 
                        int filterHeight, 
                        int filterWidth, 
                        int strideY, 
                        int strideX) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new MaxPooling2d(inputs, covered, filterHeight, filterWidth, strideY, strideX));
}

Variable& reduceMean(Variable &x, std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new ReduceMean(inputs, axis, keepDims));
}

Variable& reduceSum(Variable &x, std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new ReduceSum(inputs, axis, keepDims));
}

Variable& relu(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new ReLu(inputs));
}

Variable& reShape(Variable &x, Shape &shape) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new ReShape(inputs, shape));
}

Variable& reShape(Variable &x, std::vector<size_t> list) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new ReShape(inputs, list));
}

Variable& sigmoid(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Sigmoid(inputs));
}

Variable& softmax(Variable &x, int axis) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Softmax(inputs, axis));
}

Variable& square(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Square(inputs));
}

Variable& tanh(Variable &x) {
    std::vector<Node*> inputs = { &x };

    return x.executor->addFunction(new Tanh(inputs));
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

Variable& softmaxCrossEntropyLoss(Variable &x, Variable &y) {
    auto xshape = x.shape();
    auto yshape = y.shape();

    DEEP8_ARGUMENT_CHECK(xshape == yshape, "the shape of SoftmaxCrossEntropyLoss must be same");
    DEEP8_ARGUMENT_CHECK(1 == xshape.nDims, "the shape's ndims must be 1");

    auto pred = logSoftmax(x);

    return reduceMean(dot(linear(y, -1, 0), pred), {}, false);
}

}