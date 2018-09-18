#ifndef DEEP8_TENSORUTILS_H
#define DEEP8_TENSORUTILS_H


namespace Deep8 {

/**
  * convet the Tesnor to a Eigen Matrix, Vector or Eigen::Tensor
  */
template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> eVec(Tensor<T> &t) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(static_cast<T*>(t.pointer), t.size());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> eVec(const Tensor<T> &t) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(static_cast<T*>(t.pointer), t.shape.size());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> eVec(Tensor<T> *t) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(static_cast<T*>(t->pointer), t->size());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> eVec(const Tensor<T> *t) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(static_cast<T*>(t->pointer), t->shape.size());
}

/**
 * @brief convert to a Eigen Matrix
 * @return the Eigen::Matrix
 */
template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eMat(Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(1 == t.batch() && (2 == t.nDims() || 3 == t.nDims()), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.row(), t.col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eMat(const Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(1 == t.batch() && (2 == t.nDims() || 3 == t.nDims()), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.row(), t.col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eMat(Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(1 == t->batch() && (2 == t->nDims() || 3 == t->nDims()), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->row(), t->col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eMat(const Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(1 == t->batch() && (2 == t->nDims() || 3 == t->nDims()), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->row(), t->col());
}


/**
 * @brief convert the bth batch to a matrix, like a broadcast
 * @param b the batch number
 * @return the Matrix
 */
template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchMat(Tensor<T> &t, size_t b) {
    DEEP8_ARGUMENT_CHECK(2 == t.nDims() || 3 == t.nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer) + (b % t.batch()) * t.batchSize(), t.row(), t.col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchMat(const Tensor<T> &t, size_t b) {
    DEEP8_ARGUMENT_CHECK(2 == t.nDims() || 3 == t.nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer) + (b % t.batch()) * t.batchSize(), t.row(), t.col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eRowBatchMat(Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(2 == t.nDims() || 3 == t.nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.batch() * t.row(), t.col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eRowBatchMat(const Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(2 == t.nDims() || 3 == t.nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.batch() * t.row(), t.col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchSizeMat(Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(2 == t.nDims() || 3 == t.nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.batch(), t.batchSize());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchSizeMat(const Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(2 == t.nDims() || 3 == t.nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.batch(), t.batchSize());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchMat(Tensor<T> *t, size_t b) {
    DEEP8_ARGUMENT_CHECK(2 == t->nDims() || 3 == t->nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer) + (b % t->batch()) * t->batchSize(), t->row(), t->col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchMat(const Tensor<T> *t, size_t b) {
    DEEP8_ARGUMENT_CHECK(2 == t->nDims() || 3 == t->nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer) + (b % t->batch()) * t->batchSize(), t->row(), t->col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eRowBatchMat(Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(2 == t->nDims() || 3 == t->nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->batch() * t->row(), t->col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eRowBatchMat(const Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(2 == t->nDims() || 3 == t->nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->batch() * t->row(), t->col());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchSizeMat(Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(2 == t->nDims() || 3 == t->nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->batch(), t->batchSize());
}

template <typename T>
inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eBatchSizeMat(const Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(2 == t->nDims() || 3 == t->nDims(), "the Tensor is not a Matrix");

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->batch(), t->batchSize());
}

/**
 * convert Deep8 Tensor to Eigen Tensor with 1 dimension, as a Vector
 */
template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> eTVec(Tensor<T> &t) {
    return Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.size());
}

template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> eTVec(const Tensor<T> &t) {
    return Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>(static_cast<T*>(t.pointer), t.size());
}

template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>> eTScalar(Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(1 == t.size(), "the Tensor is not a Scalar");

    return Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>>(static_cast<T*>(t.pointer));
}

template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>> eTScalar(const Tensor<T> &t) {
    DEEP8_ARGUMENT_CHECK(1 == t.size(), "the Tensor is not a Scalar");

    return Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>>(static_cast<T*>(t.pointer));
}

template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> eTVec(Tensor<T> *t) {
    return Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->size());
}

template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> eTVec(const Tensor<T> *t) {
    return Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>(static_cast<T*>(t->pointer), t->size());
}

template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>> eTScalar(Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(1 == t->size(), "the Tensor is not a Scalar");

    return Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>>(static_cast<T*>(t->pointer));
}

template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>> eTScalar(const Tensor<T> *t) {
    DEEP8_ARGUMENT_CHECK(1 == t->size(), "the Tensor is not a Scalar");

    return Eigen::TensorMap<Eigen::Tensor<T, 0, Eigen::RowMajor>>(static_cast<T*>(t->pointer));
}


}

#endif //DEEP8_TENSORUTILS_H





















