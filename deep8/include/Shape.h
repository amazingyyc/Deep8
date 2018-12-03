#ifndef DEEP8_SHAPE_H
#define DEEP8_SHAPE_H

#include "Basic.h"
#include "Exception.h"

namespace Deep8 {

/**
 * @brief the max dimension size
 */
#define MAX_TENSOR_DIMS 5

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
	template <int NumDims>
	NVShape<NumDims> convertToNVShape() const {
		DEEP8_ARGUMENT_CHECK(NumDims >= this->numDims && NumDims > 0, "the NumDims must >= " << this->numDims);

		NVShape<NumDims> nvshape;
		nvshape.dims[0] = this->dims[0];

		for (int i = NumDims - 1, j = this->numDims - 1; i >= 1; --i, --j) {
			if (j >= 1) {
				nvshape.dims[i] = this->dims[j];
			} else {
				nvshape.dims[i] = 1;
			}
		}

		nvshape.strides[NumDims - 1] = 1;
		for (int i = NumDims - 2; i >= 0; --i) {
			nvshape.strides[i] = nvshape.strides[i + 1] * nvshape.dims[i + 1];
		}

		return nvshape;
	}

	std::string toString();
};

}

#endif //DEEP8_SHAPE_H
