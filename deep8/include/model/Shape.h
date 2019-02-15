#ifndef DEEP8_SHAPE_H
#define DEEP8_SHAPE_H

#include "basic/Basic.h"
#include "basic/Exception.h"

namespace Deep8 {

/**
 * @brief the max dimension size
 */
#define MAX_TENSOR_DIMS 4

template<int NumDims>
struct NVArray {
	int dims[NumDims];
};

/**for pass shape to CUDA*/
template<int NumDims>
struct NVShape {
	int dims[NumDims];
	int strides[NumDims];
};

/**
 * @brief the shape class store the dimension of a Tensor
 * Shape include a Batch and dims
 * The batch means how many batch this Tensor include
 * The dims means the dimension of this Tensor
 */
class Shape {
public:
	/**the batch of this Tensor*/
	size_t batch;

    /**the number of the dimension*/
    size_t nDims;

	/**the dimension*/
	size_t dims[MAX_TENSOR_DIMS];

	/**the strides of every dim*/
	size_t strides[MAX_TENSOR_DIMS];

private:
	/**update the stride by the dims*/
	void updateStrides();

public:
	Shape();

	Shape(const Shape&);
	Shape(size_t, const Shape&);

	explicit Shape(std::vector<size_t>);
	explicit Shape(size_t batch, std::vector<size_t>);

	Shape& operator = (const Shape&);

	bool operator == (const Shape&) const;
	bool operator != (const Shape&) const;

	size_t operator [] (size_t);

	void reShape(Shape&);

	bool equalExceptBatch(const Shape &);

	/**batch * dims[0] * dims[1] ... */
	size_t size() const;

	/**size() / batch*/
	size_t batchSize() const;

	size_t dim(size_t) const;
	size_t stride(size_t) const;

	size_t row() const;
	size_t col() const;

	std::string toString();
};

}

#endif //DEEP8_SHAPE_H
