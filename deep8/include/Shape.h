#ifndef DEEP8_SHAPE_H
#define DEEP8_SHAPE_H

#include "Basic.h"
#include "Exception.h"

namespace Deep8 {

/**
 * @brief the max dimension size
 */
#define MAX_TENSOR_DIMS 4

/**
 * @brief the shape class store the dimension of a Tensor
 */
class Shape {
private:
    /**
     * @brief the number of the dimension
     */
    size_t numDimension;

    /**
     * @brief store the dimension of every dim
     */
    size_t dimensions[MAX_TENSOR_DIMS];

public:
	Shape();

	Shape(std::initializer_list<size_t> list);

	Shape(std::vector<size_t> list);

	Shape(size_t batch, std::initializer_list<size_t> list);

	Shape(const Shape &otherShape);

	Shape& operator=(const Shape &otherShape);

	bool operator==(const Shape &otherShape);

    /**
     * @brief if the Shape is equal, except batch
     */
	bool equalExceptBatch(const Shape &otherShape);

	size_t batchSize() const;

	size_t size() const;

	size_t dim(size_t d) const;

	size_t nDims() const;

	size_t batch() const;

	size_t row() const;

	size_t col() const;

    /**
     * @brief reshape this Shape same to another
     * @param otherShape reshape this shape same to otherShape
     */
	void reShape(Shape &otherShape);

	void reShape(std::initializer_list<size_t> list);

    /**
     * reShape this same to other Shape, but the batch is special
     */
	void reShape(size_t batch, Shape &otherShape);
};

}

#endif //DEEP8_SHAPE_H
