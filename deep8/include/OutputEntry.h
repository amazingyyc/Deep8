#ifndef DEEP8_OUTPUTENTRY_H
#define DEEP8_OUTPUTENTRY_H

#include "Basic.h"
#include "Exception.h"

namespace Deep8 {

class Node;

class OutputEntry {
public:
	std::vector<Node*> outputs;
	std::vector<size_t> index;

	/**add a output*/
	void add(Node *node, size_t i);

	/**clear*/
	void clear();

	size_t size();
};

}

#endif
