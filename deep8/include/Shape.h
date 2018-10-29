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

	explicit Shape(std::vector<size_t> &list);
	explicit Shape(size_t batch, std::vector<size_t> &list);

	Shape(const Shape &other);

	Shape& operator=(const Shape &other);

	bool operator==(const Shape &other);

    /**@brief if the Shape is equal, except batch*/
	bool equalExceptBatch(const Shape &other);

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
	void reShape(Shape &other);
	void reShape(std::vector<size_t> list);

    /**reShape this same to other Shape, but the batch is special*/
	void reShape(size_t batch, Shape &other);

	std::string toString();
};

}

#endif //DEEP8_SHAPE_H
