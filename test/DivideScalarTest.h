#ifndef DEEP8_DIVIDESCALARTEST_H
#define DEEP8_DIVIDESCALARTEST_H

#include "DivideScalar.h"

namespace Deep8 {

TEST(DivideScalar, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto output = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
	DivideScalar<float> divideScalar(inputs, 3.0);

    std::vector<const Tensor<float>*> inputTensor = {&input};

	divideScalar.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_TRUE(std::abs(input.data()[i * 500 * 200 + j * 200 + k] / 3.0 - output.data()[i * 500 * 200 + j * 200 + k]) < 1e-6);
            }
        }
    }

    freeTensor<CPUDevice, float>(device, input);
    freeTensor<CPUDevice, float>(device, output);

    freeFakeVariable(inputVar1);

    delete device;
}

TEST(DivideScalar, backwardCPU) {
    auto device = new CPUDevice();

    auto inputValue1 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto inputGrad1  = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    auto outputValue = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto outputGrad  = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    /**create fake Add Function*/
    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
	DivideScalar<float> divideScalar(inputs, 5.0);

    zeroTensor(device, inputGrad1);

    std::vector<const Tensor<float>*> inputValues = {&inputValue1};

	divideScalar.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);

    for (int i = 0; i < 10 * 500 * 200; ++i) {
        ASSERT_TRUE(std::abs(inputGrad1.data()[i] - outputGrad.data()[i] / 5.0) < 1e-6);
    }

    freeTensor<CPUDevice, float>(device, inputValue1);
    freeTensor<CPUDevice, float>(device, inputGrad1);
    freeTensor<CPUDevice, float>(device, outputValue);
    freeTensor<CPUDevice, float>(device, outputGrad);

    freeFakeVariable(inputVar1);

    delete device;
}

#ifdef HAVE_CUDA

TEST(DivideScalar, GPU_double) {
    typedef double real;

    auto device = new GPUDevice();

    int dim0 = 10, dim1 = 500, dim2 = 200;
    real scalar = 5.5;

    auto inputPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
    auto inputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto outputPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
	auto outputGradPtr  = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto input = createTensorGPU<real>(device, inputPtr, dim0, dim1, dim2);
	auto inputGrad  = createTensorGPU<real>(device, inputGradPtr, dim0, dim1, dim2);

	auto output = createTensorGPU<real>(device, outputPtr, dim0, dim1, dim2);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, dim0, dim1, dim2);

    /**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

    zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = {&inputVar};
	DivideScalar<real> divideScalar(inputs, scalar);

    std::vector<const Tensor<real>*> inputValues = {&input};

	divideScalar.forwardGPU(inputValues, &output);
	divideScalar.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

    device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * dim0 * dim1 * dim2);
    device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * dim0 * dim1 * dim2);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_TRUE(std::abs(inputPtr[i * 500 * 200 + j * 200 + k] / scalar - outputPtr[i * 500 * 200 + j * 200 + k]) < 1e-6);
            }
        }
    }

    for (int i = 0; i < 10 * 500 * 200; ++i) {
        ASSERT_TRUE(std::abs(inputGradPtr[i] - outputGradPtr[i] / scalar) < 1e-6);
    }

    free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input);
	freeTensor(device, inputGrad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

	delete device;
}


#ifdef HAVE_HALF

TEST(DivideScalar, half_GPU) {
	typedef half real;

	auto device = new GPUDevice();

	int dim0 = 10, dim1 = 500, dim2 = 200;
	real scalar = 5.5;

	auto input = createTensorGPU<real>(device, dim0, dim1, dim2);
	auto inputGrad = createTensorGPU<real>(device, dim0, dim1, dim2);

	auto output = createTensorGPU<real>(device, dim0, dim1, dim2);
	auto outputGrad = createTensorGPU<real>(device, dim0, dim1, dim2);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar };
	DivideScalar<real> divideScalar(inputs, scalar);

	std::vector<const Tensor<real>*> inputValues = { &input };

	divideScalar.forwardGPU(inputValues, &output);
	divideScalar.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

	delete device;
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_DIVIDESCALARTEST_H
