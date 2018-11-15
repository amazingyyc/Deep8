#include "Node.h"

namespace Deep8 {

Node::Node(): type(NodeType::Unknow), id(-1) {
}

Node::Node(Node *input): type(NodeType::Unknow), id(-1) {
	this->inputs.emplace_back(input);
	input->outputs.add(this, 0);
}

Node::Node(std::vector<Node*> &in): inputs(std::move(in)), type(NodeType::Unknow), id(-1) {
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

bool Node::supportAutoBatch() {
	return false;
}

size_t Node::autoBatchCode() {
	return 0;
}

/**
 * to string
 */
std::string Node::toString() {
	return "Node: the Base class of Function and Variable";
}

}