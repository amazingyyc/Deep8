#include "Shape.h"

namespace Deep8 {


Shape::Shape() : numDims(0), dims{ 0 }, strides{ 0 } {
}

Shape::Shape(std::vector<size_t> &list) : numDims(0) {
	DEEP8_ARGUMENT_CHECK(list.size() <= MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than " << MAX_TENSOR_DIMS);

	for (auto d : list) {
		DEEP8_ARGUMENT_CHECK(0 != d, "the dim can not be 0");

		dims[numDims++] = d;
	}

	updateStrides();
}

Shape::Shape(size_t batch, std::vector<size_t> &list) : numDims(0) {
	DEEP8_ARGUMENT_CHECK(list.size() < MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than " << MAX_TENSOR_DIMS);

	dims[numDims++] = batch;

	for (auto d : list) {
		dims[numDims++] = d;
	}

	updateStrides();
}

Shape::Shape(const Shape &other) {
	numDims = other.nDims();

	for (size_t i = 0; i < numDims; ++i) {
		dims[i]    = other.dim(i);
		strides[i] = other.stride(i);
	}
}

Shape& Shape::operator=(const Shape &other) {
	numDims = other.nDims();

	for (size_t i = 0; i < numDims; ++i) {
		dims[i]    = other.dim(i);
		strides[i] = other.stride(i);
	}

	return *this;
}

bool Shape::operator==(const Shape &other) {
	if (this->numDims != other.numDims) {
		return false;
	}

	for (size_t i = 0; i < this->numDims; ++i) {
		if (this->dim(i) != other.dim(i)) {
			return false;
		}
	}

	return true;
}

bool Shape::operator!=(const Shape &other) {
	return !((*this) == other);
}

size_t Shape::operator[](size_t d) {
	return dim(d);
}

/**update the stride by the dims*/
void Shape::updateStrides() {
	DEEP8_ARGUMENT_CHECK(numDims <= MAX_TENSOR_DIMS, "the numDims is error");
	
	if (numDims > 0) {
		strides[numDims - 1] = 1;

		for (int i = (int)numDims - 2; i >= 0; --i) {
			strides[i] = strides[i + 1] * dims[i + 1];
		}
	}
}

/**
 * @brief if the Shape is equal, except batch
 */
bool Shape::equalExceptBatch(const Shape &other) {
	if (this->numDims != other.numDims) {
		return false;
	}

	for (size_t i = 1; i < this->numDims; ++i) {
		if (this->dim(i) != other.dim(i)) {
			return false;
		}
	}

	return true;
}

size_t Shape::size() const {
	if (0 == numDims) {
		return 0;
	}

	return dims[0] * strides[0];
}

size_t Shape::batchSize() const {
	DEEP8_ARGUMENT_CHECK(numDims >= 1, "the dimensions is error");

	return strides[0];
}


size_t Shape::dim(size_t d) const {
	DEEP8_ARGUMENT_CHECK(d < numDims, "the outputShape does't have dimensions " << d);

	return dims[d];
}

size_t Shape::stride(size_t d) const {
	DEEP8_ARGUMENT_CHECK(d < numDims, "the outputShape does't have dimensions " << d);

	return strides[d];
}

size_t Shape::nDims() const {
	return numDims;
}

size_t Shape::batch() const {
	DEEP8_ARGUMENT_CHECK(numDims >= 1, "the dimensions is error");

	return dims[0];
}

size_t Shape::row() const {
	DEEP8_ARGUMENT_CHECK(numDims >= 2, "the dimensions is error");

	return dims[1];
}

size_t Shape::col() const {
	DEEP8_ARGUMENT_CHECK(numDims >= 2, "the dimensions is error");

	return 2 == numDims ? 1 : dims[2];
}

/**
 * @brief reshape this Shape same to another
 * @param other reshape this shape same to otherShape
 */
void Shape::reShape(Shape &other) {
	DEEP8_ARGUMENT_CHECK(this->size() == other.size(), "the reShape operator needs this 2 Shape have the same size");

	this->numDims = other.nDims();

	for (size_t i = 0; i < numDims; ++i) {
		dims[i]    = other.dim(i);
		strides[i] = other.stride(i);
	}
}

void Shape::reShape(std::vector<size_t> &list) {
	DEEP8_ARGUMENT_CHECK(list.size() <= MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than: " << MAX_TENSOR_DIMS);

	numDims = 0;

	for (auto d : list) {
		dims[numDims++] = d;
	}

	updateStrides();
}

std::string Shape::toString() {
	std::stringstream ss;
	ss << "Rank: " << numDims;
	ss << ", Dimension: [";

	for (size_t i = 0; i < numDims; ++i) {
		ss << dims[i] << ", ";
	}

	ss << "].";

	return ss.str();
}

}