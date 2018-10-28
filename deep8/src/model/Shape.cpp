#include "Shape.h"

namespace Deep8 {


Shape::Shape() : numDimension(0), dimensions{ 0 } {
}

Shape::Shape(std::initializer_list<size_t> list) : numDimension(0) {
	DEEP8_ARGUMENT_CHECK(list.size() <= MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than " << MAX_TENSOR_DIMS);

	for (auto d : list) {
		dimensions[numDimension++] = d;
	}
}

Shape::Shape(std::vector<size_t> list) : numDimension(0) {
	DEEP8_ARGUMENT_CHECK(list.size() <= MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than " << MAX_TENSOR_DIMS);

	for (auto d : list) {
		DEEP8_ARGUMENT_CHECK(0 != d, "the dim can not be 0");

		dimensions[numDimension++] = d;
	}
}

Shape::Shape(size_t batch, std::initializer_list<size_t> list) : numDimension(0) {
	DEEP8_ARGUMENT_CHECK(list.size() < MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than " << MAX_TENSOR_DIMS);

	dimensions[numDimension++] = batch;

	for (auto d : list) {
		dimensions[numDimension++] = d;
	}
}

Shape::Shape(const Shape &otherShape) {
	numDimension = otherShape.nDims();

	for (size_t i = 0; i < numDimension; ++i) {
		dimensions[i] = otherShape.dim(i);
	}
}

Shape& Shape::operator=(const Shape &otherShape) {
	numDimension = otherShape.nDims();

	for (size_t i = 0; i < numDimension; ++i) {
		dimensions[i] = otherShape.dim(i);
	}

	return *this;
}

bool  Shape::operator==(const Shape &otherShape) {
	if (this->numDimension != otherShape.numDimension) {
		return false;
	}

	for (size_t i = 0; i < this->numDimension; ++i) {
		if (this->dim(i) != otherShape.dim(i)) {
			return false;
		}
	}

	return true;
}

/**
 * @brief if the Shape is equal, except batch
 */
bool Shape::equalExceptBatch(const Shape &otherShape) {
	if (this->numDimension != otherShape.numDimension) {
		return false;
	}

	for (size_t i = 1; i < this->numDimension; ++i) {
		if (this->dim(i) != otherShape.dim(i)) {
			return false;
		}
	}

	return true;
}

size_t Shape::batchSize() const {
	DEEP8_ARGUMENT_CHECK(numDimension >= 1, "the dimensions is error");

	size_t ret = 1;

	for (size_t i = 1; i < numDimension; ++i) {
		ret *= dimensions[i];
	}

	return ret;
}

size_t Shape::size() const {
	if (0 == numDimension) {
		return 0;
	}

	size_t ret = 1;

	for (size_t i = 0; i < numDimension; ++i) {
		ret *= dimensions[i];
	}

	return ret;
}

size_t Shape::dim(size_t d) const {
	DEEP8_ARGUMENT_CHECK(d < numDimension, "the outputShape does't have dimensions " << d);

	return dimensions[d];
}

size_t Shape::nDims() const {
	return numDimension;
}

size_t Shape::batch() const {
	DEEP8_ARGUMENT_CHECK(numDimension >= 1, "the dimensions is error");

	return dim(0);
}

size_t Shape::row() const {
	DEEP8_ARGUMENT_CHECK(numDimension >= 2, "the dimensions is error");

	return dim(1);
}

size_t Shape::col() const {
	DEEP8_ARGUMENT_CHECK(numDimension >= 2, "the dimensions is error");

	return 2 == numDimension ? 1 : dim(2);
}

/**
 * @brief reshape this Shape same to another
 * @param otherShape reshape this shape same to otherShape
 */
void Shape::reShape(Shape &otherShape) {
	DEEP8_ARGUMENT_CHECK(this->size() == otherShape.size(), "the reShape operator needs this 2 Shape have the same dim");

	this->numDimension = otherShape.nDims();

	for (size_t i = 0; i < numDimension; ++i) {
		dimensions[i] = otherShape.dim(i);
	}
}

void Shape::reShape(std::initializer_list<size_t> list) {
	DEEP8_ARGUMENT_CHECK(list.size() <= MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than: " << MAX_TENSOR_DIMS);

	this->numDimension = 0;
	for (auto d : list) {
		dimensions[numDimension++] = d;
	}
}

void Shape::reShape(std::vector<size_t> list) {
	DEEP8_ARGUMENT_CHECK(list.size() <= MAX_TENSOR_DIMS, "the dim of a outputShape must not bigger than: " << MAX_TENSOR_DIMS);

	this->numDimension = 0;
	for (auto d : list) {
		dimensions[numDimension++] = d;
	}
}

/**
 * reShape this same to other Shape, but the batch is special
 */
void Shape::reShape(size_t batch, Shape &otherShape) {
	reShape(otherShape);

	dimensions[0] = batch;
}

std::string Shape::toString() {
	std::stringstream ss;
	ss << "Rank: " << numDimension;
	ss << ", Dimension: [";

	for (size_t i = 0; i < numDimension; ++i) {
		ss << dimensions[i] << ", ";
	}

	ss << "].";

	return ss.str();
}

}