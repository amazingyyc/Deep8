#ifndef DEEP8_SHAPEUTILS_H
#define DEEP8_SHAPEUTILS_H

namespace Deep8 {

/**
 * @brief Broadcast the Shape ref:https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
 * in Deep8 the shape1 will match the shape2 batch and other dim will match from the tail.
 * if shape1 is {3, 2, 3} shape2 is {1, 3}
 * than the output is {3, 2, 3}
 * {3, 2, 3}
 *          --> {3, 2, 3}
 * {1, -, 3}
 * @return the Broadcast shape
 */
inline Shape broadcastShape(const Shape &shape1, const Shape &shape2) {
    DEEP8_ARGUMENT_CHECK(1 == shape1.batch || 1 == shape2.batch || shape1.batch == shape2.batch, "the batch must be 1 or equal");

    auto nDims = std::max<size_t>(shape1.nDims, shape2.nDims);

    std::vector<size_t> ret(nDims);

    for (int i = (int)nDims - 1, i1 = (int) shape1.nDims - 1, i2 = (int) shape2.nDims - 1; i >= 0; --i, --i1, --i2) {
        auto size1 = i1 >= 0 ? shape1.dim(i1) : 1;
        auto size2 = i2 >= 0 ? shape2.dim(i2) : 1;

        DEEP8_ARGUMENT_CHECK(1 == size1 || 1 == size2 || size1 == size2, "the Shape can not Broadcast");

        ret[i] = std::max<size_t>(size1, size2);
    }

    return Shape(std::max<size_t>(shape1.batch, shape2.batch), ret);
}

/**
 * broad the Shape to the MAX_TENSOR_DIMS dim like broadcastShape
 */
inline Eigen::array<int64_t, MAX_TENSOR_DIMS + 1> enlongateShapeToMaxDim(const Shape &shape) {
    DEEP8_ARGUMENT_CHECK(MAX_TENSOR_DIMS >= shape.nDims && shape.nDims >= 1, "the dim is error");

    Eigen::array<int64_t, MAX_TENSOR_DIMS + 1> dims;
    dims[0] = static_cast<int64_t>(shape.batch);

    for (int i = MAX_TENSOR_DIMS, j = (int)shape.nDims - 1; i > 0; --i, --j) {
        if (j >= 0) {
            dims[i] = static_cast<int64_t>(shape.dim(j));
        } else {
            dims[i] = 1;
        }
    }

    return dims;
}


/**generate a NVShape*/
template <int NumDims>
NVShape<NumDims> convertToNVShape(const Shape &shape) {
    DEEP8_ARGUMENT_CHECK(NumDims > shape.nDims && NumDims > 0, "the NumDims is error ");

    NVShape<NumDims> nvshape;
    nvshape.dims[0] = shape.batch;

    for (int i = NumDims - 1, j = (shape.nDims - 1); i >= 1; --i, --j) {
        if (j >= 0) {
            nvshape.dims[i] = shape.dims[j];
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


}

#endif //DEEP8_SHAPEUTILS_H
