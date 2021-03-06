#ifndef DEEP8_OUTPUTENTRY_H
#define DEEP8_OUTPUTENTRY_H

#include "basic/Basic.h"
#include "basic/Exception.h"

namespace Deep8 {

class Node;

class OutputEntry {
public:
	/**store the output and the index in output*/
	std::unordered_map<Node*, size_t> entries;

public:
	/**add a output*/
	void add(Node *node, size_t i);

	/**delete a node from output*/
	void remove(Node *node);

	/**clear*/
	void clear();

	/**the size of output*/
	size_t size();

	/**get the first output*/
	Node* first();
};

}

#endif
