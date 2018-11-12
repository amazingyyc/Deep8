#ifndef DEEP8_NODE_H
#define DEEP8_NODE_H

#include "Basic.h"
#include "Exception.h"
#include "MemoryAllocator.h"
#include "OutputEntry.h"
#include "Shape.h"
#include "MemoryPool.h"
#include "Device.h"
#include "TensorStorage.h"
#include "Tensor.h"

#include "ShapeUtils.h"
#include "TensorUtils.h"

namespace Deep8 {

enum class NodeType {
    Unknow,
    Variable,
    Function,
};

class Node {
public:
	/**store the input Node*/
    std::vector<Node*> inputs;

	/**store the output Node*/
	OutputEntry outputs;
    
	/**the output shape of forward*/
    Shape outputShape;

    /**what kind type of this Node, default is unknow*/
    NodeType type;

protected:
	explicit Node();
	explicit Node(Node *input);
	explicit Node(std::vector<Node*> &inputs);

public:
	virtual ~Node();

    /**
     * @brief for different Node the forward do different operation
     * Function Node: in Function Node forward, the Function Node will get the input Tensors from the inputs Nodes
     * and calculate the result than put the Result Tensor in the output Node, So the Function Node must have correct inputs and output Node type.
     * Variable Node: the Variable Node forward do nothing, it just contain the trained Parameter and update grad
     */
	virtual void forward();

    /**
     * @brief for different Node the backward do different operation
     * Function Node: in backward the Function Node get the grad from the output node, than update the inputs nodes grad.
     * Variable Node: do nothing, it just contain the grad and update the trained Parameter
     */
	virtual void backward();

    /**
     * to string
     */
	virtual std::string toString();
};

}

#endif //DEEP8_NODE_H
