#include "nodes/Node.h"

namespace Deep8 {

Node::Node(): type(NodeType::Unknow), id(-1), shape(), elementType(ElementType::unknown()), updateGradient(false) {
}

Node::Node(Node *input): type(NodeType::Unknow), id(-1), shape(), elementType(ElementType::unknown()), updateGradient(false) {
	this->inputs.emplace_back(input);
	input->outputs.add(this, 0);
}

Node::Node(std::vector<Node*> &in): inputs(in), type(NodeType::Unknow), id(-1), shape(), elementType(ElementType::unknown()), updateGradient(false) {
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

int Node::supportAutoBatch() {
	return -1;
}

size_t Node::autoBatchCode() {
	return 0;
}

/**
 * return the inputs's index that can be auto batched
 */
std::vector<size_t> Node::autoBatchIndexes() {
	return std::vector<size_t>();
}

/**
 * clone current node for auto batch
 */
Node* Node::autoBatchClone(std::vector<Node*> &inputs) {
	return nullptr;
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
Shape Node::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
	size_t size = 0;

	for (auto item : shapes) {
		size += item.size();
	}

	std::vector<size_t> vec({size});

	return Shape(1, vec);
}

/**
 * to string
 */
std::string Node::toString() {
	return "Node: the Base class of Function and Variable";
}

}