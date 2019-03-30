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

class Executor;

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

	/**maybe empty*/
	std::string name;

	/**the executor maybe nullptr*/
	Executor *executor;

	/**store the input Node*/
    std::vector<Node*> inputs;

	/**store the output Node*/
	OutputEntry outputs;

protected:
	explicit Node();
	explicit Node(Node *input);
	explicit Node(std::vector<Node*> &inputs);

	explicit Node(int64_t id, std::string name, Executor *exe);
	explicit Node(int64_t id, std::string name, Executor *exe, Node *input);
	explicit Node(int64_t id, std::string name, Executor *exe, std::vector<Node*> &inputs);

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
     * to string
     */
	virtual std::string toString();
};

}

#endif
