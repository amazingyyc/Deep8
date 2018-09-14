#ifndef DEEP8_AVGPOOLING2DTEST_H
#define DEEP8_AVGPOOLING2DTEST_H

#include "AvgPooling2d.h"

namespace Deep8 {

TEST(AvgPooling2d, forwardCPU_float) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, size_t(1), size_t(32), size_t(32), size_t(64));
    auto output = createTensor<CPUDevice, float>(device, size_t(1), size_t(16), size_t(16), size_t(64));

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device, {1, 32, 32, 64});

    std::vector<Node*> inputs = {&inputVar1};
    AvgPooling2d<float> vagPooling(inputs, true, 3, 3, 2, 2);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    vagPooling.forwardCPU(inputTensor, &output);

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            int inY = 2 * y;
            int inX = 2 * x;

            for (int c = 0; c < 64; ++c) {
                float sum = 0;

                for (int yy = 0; yy < 3; ++yy) {
                    for (int xx = 0; xx < 3; ++xx) {
                        if (inY + yy < 32 && inX + xx < 32) {
                            sum += input.data()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + c];
                        }
                    }
                }

                ASSERT_EQ(output.data()[y * 16 * 64 + x * 64 + c], sum / float(9.0));
            }
        }
    }

    freeTensor(device, input);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);

    delete device;
}

TEST(AvgPooling2d, backwardCPU_float) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, float>(device, size_t(1), size_t(32), size_t(32), size_t(64));
	auto inputGrad = createTensor<CPUDevice, float>(device, size_t(1), size_t(32), size_t(32), size_t(64));

    auto outputValue = createTensor<CPUDevice, float>(device, size_t(1), size_t(16), size_t(16), size_t(64));
    auto outputGrad  = createTensor<CPUDevice, float>(device, size_t(1), size_t(16), size_t(16), size_t(64));

    zeroTensor(device, inputGrad);

    auto inputVar = createFakeVariable<CPUDevice, float>(device, {1, 32, 32, 64});

    std::vector<Node*> inputs = {&inputVar};
    AvgPooling2d<float> avgPooling2d(inputs, false, 2, 2, 2, 2);

    std::vector<const Tensor<float>*> inputTensor = {&inputValue};

    avgPooling2d.backwardCPU(inputTensor, &outputValue, &outputGrad, 0, &inputGrad);

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            int inY = y * 2;
            int inX = x * 2;

            for (int c = 0; c < 64; ++c) {
                for (int yy = 0; yy < 2; ++yy) {
                    for (int xx = 0; xx < 2; ++xx) {
                        if (inY + yy < 32 && inX + xx < 32) {
                            auto inGrad  = inputGrad.data()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + c];
                            auto outGrad = outputGrad.data()[y * 16 * 64 + x * 64 + c];

                            ASSERT_EQ(inGrad, outGrad * float(0.25));
                        }
                    }
                }
            }
        }
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar);

    delete device;
}


#ifdef HAVE_CUDA

TEST(AvgPooling2d, forwardGPU_float) {
    auto device = new GPUDevice();

    auto cpuInputPtr  = (float*)malloc(sizeof(float) * 1 * 32 * 32 * 64);
    auto cpuOutputPtr = (float*)malloc(sizeof(float) * 1 * 16 * 16 * 64);

    auto input  = createTensorGPU<float>(device, cpuInputPtr, 1, 32, 32, 64);
	auto output = createTensorGPU<float>(device, cpuOutputPtr, 1, 16, 16, 64);

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device, { 1, 32, 32, 64 });

	std::vector<Node*> inputs = { &inputVar1 };
    AvgPooling2d<float> vagPooling(inputs, true, 3, 3, 2, 2);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    vagPooling.forwardGPU(inputTensor, &output);

    device->copyFromGPUToCPU(output.pointer, cpuOutputPtr, sizeof(float) * 1 * 16 * 16 * 64);

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            int inY = 2 * y;
            int inX = 2 * x;

            for (int c = 0; c < 64; ++c) {
                float sum = 0;

                for (int yy = 0; yy < 3; ++yy) {
                    for (int xx = 0; xx < 3; ++xx) {
                        if (inY + yy < 32 && inX + xx < 32) {
                            sum += cpuInputPtr[(inY + yy) * 32 * 64 + (inX + xx) * 64 + c];
                        }
                    }
                }

                ASSERT_EQ(cpuOutputPtr[y * 16 * 64 + x * 64 + c], sum / float(9));
            }
        }
    }

    freeTensor(device, input);
	freeTensor(device, output);

	freeFakeVariable(inputVar1);

	free(cpuInputPtr);
	free(cpuOutputPtr);

    delete device;
}

TEST(AvgPooling2d, backwardGPU_float) {
    typedef float real;

    auto device = new GPUDevice();

    auto cpuInputValuePtr = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);
	auto cpuInputGradPtr  = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);

	auto cpuOutputValuePtr = (real*)malloc(sizeof(real) * 1 * 16 * 16 * 64);
	auto cpuOutputGradPtr  = (real*)malloc(sizeof(real) * 1 * 16 * 16 * 64);

	auto inputValue = createTensorGPU<real>(device, cpuInputValuePtr, 1, 32, 32, 64);
	auto inputGrad  = createTensorGPU<real>(device, cpuInputGradPtr, 1, 32, 32, 64);

	auto outputValue = createTensorGPU<real>(device, cpuOutputValuePtr, 1, 16, 16, 64);
	auto outputGrad = createTensorGPU<real>(device, cpuOutputGradPtr, 1, 16, 16, 64);

    /**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device, { 1, 32, 32, 64 });

    std::vector<Node*> inputs = {&inputVar};
	AvgPooling2d<float> vagPooling(inputs, true, 3, 3, 2, 2);

	zeroTensor(device, inputGrad);

    std::vector<const Tensor<real>*> inputValues = {&inputValue};

	vagPooling.backwardGPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    device->copyFromGPUToCPU(inputGrad.pointer, cpuInputGradPtr, sizeof(float) * 1 * 32 * 32 * 64);

	auto tempinputgradptr = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);
	memset(tempinputgradptr, 0, sizeof(real) * 1 * 32 * 32 * 64);

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            int inY = y * 2;
            int inX = x * 2;

            for (int c = 0; c < 64; ++c) {

                for (int yy = 0; yy < 3; ++yy) {
                    for (int xx = 0; xx < 3; ++xx) {
                        if (inY + yy < 32 && inX + xx < 32) {
							tempinputgradptr[(inY + yy) * 32 * 64 + (inX + xx) * 64 + c] += (cpuOutputGradPtr[y * 16 * 64 + x * 64 + c] / 9.0);
                        }
                    }
                }
            }
        }
    }

	for (int i = 0; i < 32 * 32 * 64; ++i) {
		ASSERT_TRUE(abs(tempinputgradptr[i] - cpuInputGradPtr[i]) < 1e-6);
	}

	free(tempinputgradptr);

    free(cpuInputValuePtr);
	free(cpuInputGradPtr);
	free(cpuOutputValuePtr);
	free(cpuOutputGradPtr);

	freeTensor(device, inputValue);
	freeTensor(device, inputGrad);
	freeTensor(device, outputValue);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

    delete device;
}

#endif

}

#endif //DEEP8_AVGPOOLING2DTEST_H
