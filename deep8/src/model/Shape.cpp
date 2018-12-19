#include "Shape.h"

namespace Deep8 {

Shape::Shape() : batch(0), nDims(0), dims{ 0 }, strides{ 0 } {
}

Shape::Shape(const Shape &other) {
	this->batch = other.batch;
	this->nDims = other.nDims;

	for (size_t i = 0; i < this->nDims; ++i) {
		this->dims[i]    = other.dims[i];
		this->strides[i] = other.strides[i];
	}
}

Shape::Shape(size_t b, const Shape& other): batch(b) {
	this->nDims = other.nDims;

	for (size_t i = 0; i < this->nDims; ++i) {
		this->dims[i]    = other.dims[i];
		this->strides[i] = other.strides[i];
	}
}

Shape::Shape(std::vector<size_t> list): batch(1), nDims(0) {
	DEEP8_ARGUMENT_CHECK(0 < list.size() && list.size() <= MAX_TENSOR_DIMS, "the dim of a Tensor must not bigger than " << MAX_TENSOR_DIMS);

	for (auto d : list) {
		DEEP8_ARGUMENT_CHECK(0 != d, "the dim can not be 0");

		dims[nDims++] = d;
	}

	updateStrides();
}

Shape::Shape(size_t b, std::vector<size_t> list): batch(b), nDims(0) {
	DEEP8_ARGUMENT_CHECK(0 < list.size() && list.size() <= MAX_TENSOR_DIMS, "the dim of a Tensor must not bigger than " << MAX_TENSOR_DIMS);

	for (auto d : list) {
		DEEP8_ARGUMENT_CHECK(0 != d, "the dim can not be 0");

		dims[nDims++] = d;
	}

	updateStrides();
}

Shape& Shape::operator=(const Shape &other) {
	this->batch = other.batch;
	this->nDims = other.nDims;

	for (size_t i = 0; i < this->nDims; ++i) {
		this->dims[i]    = other.dims[i];
		this->strides[i] = other.strides[i];
	}

	return (*this);
}

bool Shape::operator==(const Shape &other) {
	if (this->nDims != other.nDims || this->batch != other.batch) {
		return false;
	}

	for (size_t i = 0; i < this->nDims; ++i) {
		if (this->dims[i] != other.dims[i]) {
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
	DEEP8_ARGUMENT_CHECK(nDims <= MAX_TENSOR_DIMS, "the nDims is error");
	
	if (nDims > 0) {
		strides[nDims - 1] = 1;

		for (int i = (int)nDims - 2; i >= 0; --i) {
			strides[i] = strides[i + 1] * dims[i + 1];
		}
	}
}

/**
 * @brief if the Shape is equal, except batch
 */
bool Shape::equalExceptBatch(const Shape &other) {
	if (this->nDims != other.nDims) {
		return false;
	}

	for (size_t i = 0; i < this->nDims; ++i) {
		if (this->dim(i) != other.dim(i)) {
			return false;
		}
	}

	return true;
}

size_t Shape::size() const {
	if (0 == nDims) {
		return 0;
	}

	return batch * batchSize();
}

size_t Shape::batchSize() const {
	if (0 == nDims) {
		return 0;
	}

	return dims[0] * strides[0];
}


size_t Shape::dim(size_t d) const {
	DEEP8_ARGUMENT_CHECK(d < nDims, "the Tensor does't have dimensions " << d);

	return dims[d];
}

size_t Shape::stride(size_t d) const {
	DEEP8_ARGUMENT_CHECK(d < nDims, "the Tensor does't have dimensions " << d);

	return strides[d];
}

size_t Shape::row() const {
	DEEP8_ARGUMENT_CHECK(nDims >= 1, "the dimensions is error");

	return dims[0];
}

size_t Shape::col() const {
	DEEP8_ARGUMENT_CHECK(nDims >= 1, "the dimensions is error");

	return 1 == nDims ? 1 : dims[1];
}

/**
 * @brief reshape this Shape same to another
 * @param other reshape this shape same to otherShape
 */
void Shape::reShape(Shape &other) {
	DEEP8_ARGUMENT_CHECK(this->size() == other.size(), "the reShape operator needs this 2 Shape have the same size");

	this->batch = other.batch;
	this->nDims = other.nDims;

	for (size_t i = 0; i < this->nDims; ++i) {
		this->dims[i]    = other.dims[i];
		this->strides[i] = other.strides[i];
	}
}

std::string Shape::toString() {
	std::stringstream ss;
	ss << "nDims: " << nDims;
	ss << ", Dimension: [";

	for (size_t i = 0; i < nDims; ++i) {
		ss << dims[i] << ", ";
	}

	ss << "].";

	return ss.str();
}

}