#ifndef DEEP8_VARIABLE_H
#define DEEP8_VARIABLE_H

#include "model/Executor.h"
#include "nodes/Node.h"

namespace Deep8 {

class Variable: public Node {
public:
	Tensor value;
	Tensor gradient;

    /**if update the gradient of this Variable*/
    bool updateGradient;

protected:
    explicit Variable(int64_t id, std::string name, Executor *exe);

public:
    explicit Variable(int64_t id, std::string name, Executor *exe, Tensor &v);
    explicit Variable(int64_t id, std::string name, Executor *exe, Tensor &v, Tensor &g);

    explicit Variable(int64_t id, std::string name, Executor *exe, Node *input);
	explicit Variable(int64_t id, std::string name, Executor *exe, Node *input, Tensor &v);
    explicit Variable(int64_t id, std::string name, Executor *exe, Node *input, Tensor &v, Tensor &g);

public:

    /**get the Variable shape*/
    Shape shape();

    /**get the element type*/
    ElementType elementType();

    /**get the device type*/
    DeviceType deviceType();
    
    /**if this is a scalar*/
    bool isScalar();

    /**set the Gradient to be 0*/
	void zeroGradient();

    /**set gradient to one*/
    void oneGradient();

	/**release the gradient*/
	void removeGradient();

	/** the Variable do nothing in forward and backward process*/
	void forward() override;
	void backward() override;

    /****************************************************************************/
    /**function for build computer graph*/
    /****************************************************************************/
    
    /**return a string for print value*/
    std::string valueStr();

    /**feed data to value from CPU memory*/
    Variable& feed(const void*);

    /**copy memory from value to CPU memory*/
    Variable& fetch(void*);

	Variable& constant(float scalar = 0);
	Variable& zero();
	Variable& one();
	Variable& gaussian(float mean = 0.0, float stddev = 0.01);
	Variable& positiveUnitball();
	Variable& random(float lower = 0.0, float upper = 1.0);
	Variable& uniform(float left = 0.0, float right = 1.0);
	Variable& assign(Variable& v);

    Variable& add(Variable &y);
    Variable& minus(Variable &y);
    Variable& multiply(Variable &y);
    Variable& divide(Variable &y);

    Variable& addConstant(float c);
    Variable& minusConstant(float c);
    Variable& multiplyConstant(float c);
    Variable& divideConstant(float c);

    Variable& abs();

    Variable& avgPooling2d(bool covered = true, 
                           int filterHeight = 1, 
                           int filterWidth = 1, 
                           int strideY = 1, 
                           int strideX = 1);
    
    Variable& conv2d(Variable &filter,
                    bool covered = true, 
                    int strideY = 1, 
                    int strideX = 1, 
                    int dilationY = 1, 
                    int dilationX = 1);

    Variable& crossEntropy(Variable &y);

    Variable& deConv2d( Variable &filter,
                        bool covered = false, 
                        int strideY = 1, 
                        int strideX = 1);

    Variable& dot(Variable &y);
    Variable& exp();
    Variable& l1Distance(Variable &y);
    Variable& l1Norm();
    Variable& l2Distance(Variable &y);
    Variable& l2Norm();
    Variable& linear(float a = 1, float b = 0);
    Variable& log();
    Variable& logSoftmax(int axis = -1);
    Variable& lRelu(float a);

    Variable& matrixMultiply(Variable &y);

    Variable& maxPooling2d( bool covered = false, 
                            int filterHeight = 1, 
                            int filterWidth = 1, 
                            int strideY = 1, 
                            int strideX = 1);

    Variable& maxPooling2dWithIndex(Variable& index,
                                    bool covered = false, 
                                    int filterHeight = 1, 
                                    int filterWidth = 1, 
                                    int strideY = 1, 
                                    int strideX = 1);
    
    Variable& maxUnPooling2d(Variable& index,
                            bool covered = false, 
                            int filterHeight = 1, 
                            int filterWidth = 1, 
                            int strideY = 1, 
                            int strideX = 1);
    
    Variable& pRelu(Variable &p);

    Variable& reduceMean(std::vector<int> axis = {-1}, bool keepDims = true);
    Variable& reduceSum(std::vector<int> axis = {-1}, bool keepDims = true);
    Variable& relu();
    Variable& reShape(Shape &shape);
    Variable& reShape(std::vector<size_t> list);
    Variable& sigmoid();
    Variable& softmax(int axis = -1);
    Variable& sqrt();
    Variable& square();
    Variable& tanh();

    Variable& l1Loss(Variable &y);
    Variable& l1NormLoss();
    Variable& l2Loss(Variable &y);
    Variable& l2NormLoss();
    Variable& softmaxCrossEntropyLoss(Variable &y);
};

}

#endif //DEEP8_VARIABLE_H
