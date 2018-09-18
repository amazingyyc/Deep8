#ifndef DEEP8_RESHAPE_H
#define DEEP8_RESHAPE_H

namespace Deep8 {

template <typename T>
class ReShape: public Function<T> {
public:
    Shape reShape;

    explicit ReShape(std::vector<Node *> &inputs, Shape &shape): Function<T>(inputs), reShape(shape) {
        this->shared = true;
        check();
    }

    explicit ReShape(std::vector<Node *> &inputs, std::initializer_list<size_t> shape): Function<T>(inputs), reShape(shape) {
        this->shared = true;
        check();
    }

    explicit ReShape(std::vector<Node *> &inputs, std::vector<size_t> shape): Function<T>(inputs), reShape(shape) {
        this->shared = true;
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
        DEEP8_ARGUMENT_CHECK(reShape.nDims() < MAX_TENSOR_DIMS, "the reShape is error");
        DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.size() == reShape.size(), "the input's Shape size must be equal to reShape size");

        this->outputShape = reShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {

    }

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {

    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {

    }

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
    }

public:
    void forward() override {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->output->type, "the output must be Variable type");

		auto output = static_cast<Variable<T>*>(this->output);
		auto input  = static_cast<Variable<T>*>(this->inputs[0]);

		DEEP8_ARGUMENT_CHECK(this->outputShape == output->value.shape, "the output shape is error");

		output->updateGradient = input->updateGradient;
		output->shared = true;
		
		output->value.shape   = this->outputShape;
		output->value.pointer = input->value.pointer;
		output->value.device  = input->value.device;
		
		output->gradient.shape   = this->outputShape;
		output->gradient.pointer = input->gradient.pointer;
		output->gradient.device  = input->gradient.device;
    }

    void backward() override {
    }
};


}

#endif //DEEP8_RESHAPE_H
