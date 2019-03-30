#include "nodes/Node.h"

namespace Deep8 {

Node::Node():type(NodeType::Unknow), id(-1), name(), executor(nullptr) {
}

Node::Node(Node *input): type(NodeType::Unknow), id(-1), name(), executor(nullptr) {
    this->inputs.emplace_back(input);
    input->outputs.add(this, 0);
}

Node::Node(std::vector<Node*> &in): type(NodeType::Unknow), inputs(in), id(-1), name(), executor(nullptr) {
	for (size_t i = 0; i < inputs.size(); ++i) {
		inputs[i]->outputs.add(this, i);
	}
}

Node::Node(int64_t i, std::string n, Executor *exe): type(NodeType::Unknow), id(i), name(n), executor(exe) {
}

Node::Node(int64_t i, std::string n, Executor *exe, Node *input): type(NodeType::Unknow), id(i), name(n), executor(exe) {
    this->inputs.emplace_back(input);
    input->outputs.add(this, 0);
}

Node::Node(int64_t i, std::string n, Executor *exe, std::vector<Node*> &in): type(NodeType::Unknow), inputs(in), id(i), name(n), executor(exe)  {
	for (size_t i = 0; i < inputs.size(); ++i) {
		inputs[i]->outputs.add(this, i);
	}
}

Node::~Node() {
}

void Node::forward() {
	DEEP8_RUNTIME_ERROR("Can not call this function from Node");
}

/**
 * @brief for different Node the backward do different operation
 * Function Node: in backward the Function Node get the grad from the output node, than update the inputs nodes grad.
 * Variable Node: do nothing, it just contain the grad and update the trained Parameter
 */
void Node::backward() {
	DEEP8_RUNTIME_ERROR("Can not call this function from Node");
}

/**
 * to string
 */
std::string Node::toString() {
	return "Node: the Base class of Function and Variable";
}

}