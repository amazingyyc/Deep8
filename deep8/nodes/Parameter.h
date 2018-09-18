#ifndef DEEP8_PARAMETER_H
#define DEEP8_PARAMETER_H

namespace Deep8 {

/**
 * the Parameter Node is a special Variable Node that need to be trained
 */
template <typename T>
class Parameter: public Variable<T> {
protected:
    explicit Parameter(): Variable<T>() {
    }

    explicit Parameter(Tensor<T> &value): Variable<T>(value) {
    }

public:
    explicit Parameter(Tensor<T> &value, Tensor<T> &gradient): Variable<T>(value, gradient) {
        check();
    }

protected:
    void check() override {
        DEEP8_ARGUMENT_CHECK(this->value.device->type == this->gradient.device->type, "the values and gradient must be the same type");
        DEEP8_ARGUMENT_CHECK(this->value.shape == this->gradient.shape, "the shape if Value and Gradient must be same");

        this->outputShape = this->value.shape;
    }
};

}

#endif //DEEP8_PARAMETER_H
