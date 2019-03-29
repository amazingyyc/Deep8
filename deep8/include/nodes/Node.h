#ifndef DEEP8_NODE_H
#define DEEP8_NODE_H

#include "basic/Basic.h"
#include "basic/Exception.h"
#include "model/Executor.h"
#include "model/MemoryAllocator.h"
#include "model/OutputEntry.h"
#include "model/Shape.h"
#include "model/MemoryPool.h"
#include "model/Device.h"
#include "model/TensorStorage.h"
#include "model/Tensor.h"
#include "utils/ShapeUtils.h"

namespace Deep8 {

/**
 * the Node type for now 2 type: Variable and Function
 */
enum class NodeType {
    Unknow,
    Variable,
    Function,
};

class Node {
public:
    /**what kind type of this Node, default is unknow*/
    NodeType type;

	/**the Node unique id*/
	int64_t id;

    /**the node have a unique name*/
    std::string name;

	/**store the input Node*/
    std::vector<Node*> inputs;

	/**store the output Node*/
	OutputEntry outputs;

protected:
	explicit Node(int64_t id, std::string name);
	explicit Node(int64_t id, std::string name, Node *input);
	explicit Node(int64_t id, std::string name, std::vector<Node*> &inputs);

public:
	virtual ~Node();

    /**
     * for different Node the forward do different operation
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
	 * return < 0, does not support autobatch
	 * return >= 0, means the Node can be batched
	 */
	virtual int supportAutoBatch();

	/**
	 * return a hashcode to do auto batch
	 * default is 0
	 */
	virtual size_t autoBatchCode();

	/**
	 * return the inputs's index that can be auto batched
	 */
	virtual std::vector<size_t> autoBatchIndexes();

	/**
	 * clone current node for auto batch
	 */
	virtual Node* autoBatchClone(std::vector<Node*> &);

	/**
	 * return the inputs[index]'s shape if it is be batched together.
	 * the shapes is the inputs[index]'s shape that will be batched.
	 */
	virtual Shape autoBatchShape(size_t index, std::vector<Shape> &shapes);

    /**
     * to string
     */
	virtual std::string toString();
};

}

#endif
