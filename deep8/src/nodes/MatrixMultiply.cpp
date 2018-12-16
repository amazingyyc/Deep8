#include "MatrixMultiply.h"
#include "AutoBatchCodeHelper.h"

namespace Deep8 {

template <typename T>
MatrixMultiply<T>::MatrixMultiply(std::vector<Node *> &inputs) : Function<T>(inputs) {
    check();
}

template <typename T>
void MatrixMultiply<T>::check() {
    Function <T> ::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs dim must be 2");

    auto xShape = this->inputs[0]->outputShape;
    auto yShape = this->inputs[1]->outputShape;

    DEEP8_ARGUMENT_CHECK(
            xShape.batch == yShape.batch || 1 == xShape.batch || 1 == yShape.batch,
            "the batch of input is error");
    DEEP8_ARGUMENT_CHECK((1 == xShape.nDims || 2 == xShape.nDims) &&
                         (1 == yShape.nDims || 2 == yShape.nDims),
                         "the inputs dimensions is error");
    DEEP8_ARGUMENT_CHECK(xShape.col() == yShape.row(),
                         "the col of input1 must same to the row of input2");

    if (1 == yShape.col()) {
        this->outputShape = Shape(std::max<size_t>(xShape.batch, yShape.batch), {xShape.row()});
    } else {
        this->outputShape = Shape(std::max<size_t>(xShape.batch, yShape.batch), {xShape.row(), yShape.col()});
    }
}

/**
 * 2 type MatrixMultiply can support autobatch
 * for z1 = x1 * y1, z2 = x2 * y2
 * 1: the x1 is the same with x2 and batch is 1, and the the y1 and y2's shape is same except the batch, and the col of y1 and y2 is 1.
 * 2: the y1 is same with y2 and batch is 1, the x1 and x2's col is same
 *
 * in some case the MatrixMultiply function may hit this 2 type at same time.
 * then the type 1 will be selected
 */
template <typename T>
int MatrixMultiply<T>::supportAutoBatch() {
    auto &xshape = this->inputs[0]->outputShape;
    auto &yshape = this->inputs[1]->outputShape;

    if (1 == xshape.batch && 1 == yshape.col()) {
        return 1;
    } else if (1 == yshape.batch) {
        return 0;
    }

    return -1;
}

/**
 * for 1 support type: the hashcode combined by the FunctionType 
 */
template <typename T>
size_t MatrixMultiply<T>::autoBatchCode() {
    auto index = supportAutoBatch();

    if (0 == index) {
        AutoBatchCodeHelper helper;

        helper.functionType(FunctionType::MatrixMultiply);

        helper.input0Begin();
        helper.col(this->inputs[0]->outputShape.col());
        helper.input0End();

        helper.input1Begin();
        helper.nodeId(this->inputs[1]->id);
        helper.input1End();

        return helper.autoBatchCode();
    } else if (1 == index) {
        AutoBatchCodeHelper helper;

        helper.functionType(FunctionType::MatrixMultiply);

        helper.input0Begin();
        helper.nodeId(this->inputs[0]->id);
        helper.input0End();

        helper.input1Begin();
        helper.row(this->inputs[1]->outputShape.row());
        helper.row(this->inputs[1]->outputShape.col());
        helper.input1End();

        return helper.autoBatchCode();
    }

    DEEP8_RUNTIME_ERROR("this node does not suppot autobatch");
}

/**
 * return the inputs[index]'s shape if it can be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
template <typename T>
Shape MatrixMultiply<T>::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
	DEEP8_ARGUMENT_CHECK(shapes.size() > 1, "the batched inputs's size must > 1");

    if (0 == index) {
        size_t row = 0;
        auto col = shapes[0].col();

        for (auto item : shapes) {
            DEEP8_ARGUMENT_CHECK(col == item.col(), "the batched shape is error");

            row += item.batch * item.row();
        }

        return Shape(1, { row, col });
    } else if (1 == index) {
        size_t batch = 0;
        auto row = shapes[0].row();

        for (auto item : shapes) {
            DEEP8_ARGUMENT_CHECK(1 == item.col() && row == item.row(), "the batched shape is error");

            batch += item.batch;
        }

        return Shape(batch, { row, 1 });
    }

    DEEP8_RUNTIME_ERROR("the index is error");
}

/**
 * the MatrixMultiply's autobatch index only support 1
 */
template <typename T>
std::vector<size_t> MatrixMultiply<T>::autoBatchIndexes() {
    auto index = supportAutoBatch();

    if (0 == index) {
        return std::vector<size_t>({ 0 });
    } else if (1 == index) {
        return std::vector<size_t>({ 1 });
    }

	DEEP8_RUNTIME_ERROR("this node does not support auto batch");
}

/**
 * clone current node for auto batch
 */
template <typename T>
Node* MatrixMultiply<T>::autoBatchClone(std::vector<Node*> &inputs) {
	return new MatrixMultiply<T>(inputs);
}

template <typename T>
void MatrixMultiply<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output)  {
    auto x = inputs[0];
    auto y = inputs[1];
    auto z = output;

    if (1 == y->batch()) {
        eRowBatchMat(z).noalias() = eRowBatchMat(x) * eMat(y);
    } else if (1 == x->batch() && 1 == y->col()) {
        eBatchSizeMat(z).noalias() = eBatchSizeMat(y) * eMat(x).transpose();
    } else {
        DEEP8_ARGUMENT_CHECK(1 == x->batch() || x->batch() == y->batch(), "the inputs batch error");
        DEEP8_ARGUMENT_CHECK(std::max<size_t>(x->batch(), y->batch()) == z->batch(), "the output batch is error");

        for (size_t b = 0; b < z->batch(); ++b) {
            eBatchMat(z, b).noalias() = eBatchMat(x, b) * eBatchMat(y, b);
        }
    }
}

template <typename T>
void MatrixMultiply<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    auto x  = inputs[0];
    auto y  = inputs[1];
    auto z  = output;
    auto dz = outputGradient;
    
    if (0 == index) {
        auto dx = iGradient;

        /**
         * for a MatrixMultiply Z = X * Y, index is 0 means calculate the grad for X
         * dx = dz * transpose(y)
         */
        if (1 == y->batch()) {
            eRowBatchMat(dx).noalias() += eRowBatchMat(dz) * eMat(y).transpose();
        } else if (1 == x->batch() && 1 == y->col()) {
            eMat(dx).noalias() += eBatchSizeMat(dz).transpose() * eBatchSizeMat(y);
        } else {
            for (size_t b = 0; b < outputGradient->batch(); ++b) {
                eBatchMat(dx, b).noalias() += eBatchMat(dz, b) * eBatchMat(y, b).transpose();
            }
        }
    } else if (1 == index) {
        auto dy = iGradient;

        /**
         * for a MatrixMultiply Z = X * Y, index is 1 means calculate the grad for Y
         * dy = transpose(x) * dz
         */
        if (1 == dy->batch()) {
            eMat(dy).noalias() += eRowBatchMat(x).transpose() * eRowBatchMat(dz);
        } else if (1 == x->batch() && 1 == y->col()) {
            eBatchSizeMat(dy).noalias() += eBatchSizeMat(dz) * eMat(x);
        } else {
            for (size_t b = 0; b < dz->batch(); ++b) {
                eBatchMat(dy, b).noalias() += eBatchMat(x, b).transpose() * eBatchMat(dz, b);
            }
        }
    }
}

DEEP8_RE_DECLARATION_HALF_FUNC(MatrixMultiply)
DEEP8_DECLARATION_INSTANCE(MatrixMultiply)

}