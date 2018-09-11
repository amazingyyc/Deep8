#ifndef DEEP8_NODE_H
#define DEEP8_NODE_H

#include <utility>
#include <vector>

#include "Tensor.h"
#include "Device.h"

namespace Deep8 {

enum class NodeType {
    Unknow,
    Variable,
    Function,
};

class Node {
public:
    /**
     * @brief the input nodes, include the pointers of the input of this Node
     * the Function Node may be have many inputs, but the Variable Only have 0 / 1 input
     * like a graph z = x + y
     * the graph is below:
     * (x)------
               |
               -----> [+]---->(z)
               |
       (y)------

        it include 3 Variables Node x, y, z and 1 Function Node add.
        the (x), (y) Node don't have inputs but have 1 output is [Add] Function Node (But the Variable don't output any data)
        the (z) Node have 1 input is [Add] Node But it does not have any output
        the [Add] Node have 2 input is (x), (y) and 1 output (z)
        when forward calculate graph, the (x), (y), (z) does not do anything.
        the [Add] function add the inputs (x), (y) and output the result to (z)
     */
    std::vector<Node*> inputs;

    /**
     * @brief the output shape of forward
     */
    Shape outputShape;

    /**
     * if a Variable or Function shared is true means the output Variable shared the memory with the input Variable
     * default is false
     */
    bool shared;

    /**
     * @brief what kind type of this Node, default is unknow
     */
    NodeType type;

protected:
    explicit Node(): outputShape(), shared(false), type(NodeType::Unknow) {
    }

    explicit Node(std::vector<Node*> &inputs): inputs(std::move(inputs)), outputShape(), shared(false), type(NodeType::Unknow) {
    }

public:
    virtual ~Node() = default;

    /**
     * @brief for different Node the forward do different operation
     * Function Node: in Function Node forward, the Function Node will get the input Tensors from the inputs Nodes
     * and calculate the result than put the Result Tensor in the output Node, So the Function Node must have correct inputs and output Node type.
     * Variable Node: the Variable Node forward do nothing, it just contain the trained Parameter and update grad
     */
    virtual void forward() = 0;

    /**
     * @brief for different Node the backward do different operation
     * Function Node: in backward the Function Node get the grad from the output node, than update the inputs nodes grad.
     * Variable Node: do nothing, it just contain the grad and update the trained Parameter
     */
    virtual void backward() = 0;
};

}

#endif //DEEP8_NODE_H
