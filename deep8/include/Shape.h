#ifndef DEEP8_SHAPE_H
#define DEEP8_SHAPE_H

#include "Basic.h"
#include "Exception.h"

namespace Deep8 {

/**
 * @brief the max dimension size
 */
#define MAX_TENSOR_DIMS 5

/**for pass shape to CUDA*/
struct NVShape {
	int dims[MAX_TENSOR_DIMS];
	int strides[MAX_TENSOR_DIMS];
};

/**
 * @brief the shape class store the dimension of a Tensor
 */
class Shape {
private:
    /**the number of the dimension*/
    size_t numDims;

	/**the dimension*/
	size_t dims[MAX_TENSOR_DIMS];

	/**the strides of every dim*/
	size_t strides[MAX_TENSOR_DIMS];

private:
	/**update the stride by the dims*/
	void updateStrides();

public:
	Shape();

	explicit Shape(std::vector<size_t> &list);
	explicit Shape(size_t batch, std::vector<size_t> &list);

	Shape(const Shape &other);

	Shape& operator = (const Shape &other);

	bool operator == (const Shape &other);
	bool operator != (const Shape &other);

	size_t operator[](size_t d);

    /**@brief if the Shape is equal, except batch*/
	bool equalExceptBatch(const Shape &other);

	size_t size() const;
	size_t batchSize() const;

	size_t nDims() const;
	size_t dim(size_t d) const;
	size_t stride(size_t d) const;
	
	size_t batch() const;
	size_t row() const;
	size_t col() const;

	void reShape(Shape &other);
	void reShape(std::vector<size_t> &list);

	/**generate a NVShape*/
	NVShape convertToNVShape();

	std::string toString();
};

}

#endif //DEEP8_SHAPE_H
