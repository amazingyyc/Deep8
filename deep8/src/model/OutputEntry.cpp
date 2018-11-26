#include "Node.h"
#include "OutputEntry.h"

namespace Deep8 {

void OutputEntry::add(Node *node, size_t i) {
	DEEP8_ARGUMENT_CHECK(nullptr != node, "can not add nullptr");
	DEEP8_ARGUMENT_CHECK(outputs.find(node) != outputs.end(), "can not add duplicate node");

	outputs[node] = i;
}

void OutputEntry::clear() {
	outputs.clear();
}

void OutputEntry::remove(Node *node) {
	DEEP8_ARGUMENT_CHECK(nullptr != node, "can not remove nullptr");

	outputs.erase(node);
}

size_t OutputEntry::size() {
	return outputs.size();
}

/**get the first output*/
Node* OutputEntry::first() {
	DEEP8_ARGUMENT_CHECK(this->size() >=0, "the outputs is empty");
	return outputs.begin()->first;
}

}