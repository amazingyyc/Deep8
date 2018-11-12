#include "Node.h"
#include "OutputEntry.h"

namespace Deep8 {

void OutputEntry::add(Node *node, size_t i) {
	outputs.emplace_back(node);
	index.emplace_back(i);
}

void OutputEntry::clear() {
	outputs.clear();
	index.clear();
}

size_t OutputEntry::size() {
	DEEP8_ARGUMENT_CHECK(outputs.size() == index.size(), "the size must be equal");

	return outputs.size();
}

}