#ifndef DEEP8_AVGPOOLING2DTEST_H
#define DEEP8_AVGPOOLING2DTEST_H

#include "nodes/AvgPooling2d.h"

namespace Deep8 {

TEST(AvgPooling2d, forwardCPU_float) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), 1, {32, 32, 64});
    auto output = createTensor(device, ElementType::from<float>(), 1, {16, 16, 64});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>(), 1, {32, 32, 64});

    std::vector<Node*> inputs = {&inputVar1};
    AvgPooling2d vagPooling(inputs, true, 3, 3, 2, 2);

    std::vector<const Tensor*> inputTensor = {&input};

    vagPooling.forward(inputTensor, &output);

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            int inY = 2 * y;
            int inX = 2 * x;

            for (int c = 0; c < 64; ++c) {
                float sum = 0;

                for (int yy = 0; yy < 3; ++yy) {
                    for (int xx = 0; xx < 3; ++xx) {
                        if (inY + yy < 32 && inX + xx < 32) {
                            sum += input.data<float>()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + c];
                        }
                    }
                }

                ASSERT_EQ(output.data<float>()[y * 16 * 64 + x * 64 + c], sum / float(9.0));
            }
        }
    }
}

TEST(AvgPooling2d, backwardCPU_float) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<float>(), 1, {32, 32, 64});
	auto inputGrad = createTensor(device, ElementType::from<float>(), 1, {32, 32, 64});

    auto outputValue = createTensor(device, ElementType::from<float>(), 1, {16, 16, 64});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 1, {16, 16, 64});

    zeroTensor(device, inputGrad);

    auto inputVar = createFakeVariable(device, ElementType::from<float>(), 1, {32, 32, 64});

    std::vector<Node*> inputs = {&inputVar};
    AvgPooling2d avgPooling2d(inputs, false, 2, 2, 2, 2);

    std::vector<const Tensor*> inputTensor = {&inputValue};

    avgPooling2d.backward(inputTensor, &outputValue, &outputGrad, 0, &inputGrad);

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            int inY = y * 2;
            int inX = x * 2;

            for (int c = 0; c < 64; ++c) {
                for (int yy = 0; yy < 2; ++yy) {
                    for (int xx = 0; xx < 2; ++xx) {
                        if (inY + yy < 32 && inX + xx < 32) {
                            auto inGrad  = inputGrad.data<float>()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + c];
                            auto outGrad = outputGrad.data<float>()[y * 16 * 64 + x * 64 + c];

                            ASSERT_EQ(inGrad, outGrad * float(0.25));
                        }
                    }
                }
            }
        }
    }
}


#ifdef HAVE_CUDA

TEST(AvgPooling2d, forwardGPU_float) {
	GPUDevice device;

    auto cpuInputPtr  = (float*)malloc(sizeof(float) * 1 * 32 * 32 * 64);
    auto cpuOutputPtr = (float*)malloc(sizeof(float) * 1 * 16 * 16 * 64);

    auto input  = createTensor(device, cpuInputPtr, ElementType::from<float>(), 1, {32, 32, 64});
    auto output = createTensor(device, cpuOutputPtr, ElementType::from<float>(), 1, {16, 16, 64});

	auto inputVar1 = createFakeVariable(device, ElementType::from<float>(), 1, {32, 32, 64 });

	std::vector<Node*> inputs = { &inputVar1 };
    AvgPooling2d vagPooling(inputs, true, 3, 3, 2, 2);

    std::vector<const Tensor*> inputTensor = {&input};

    vagPooling.forward(inputTensor, &output);

    device.copyFromGPUToCPU(output.raw(), cpuOutputPtr, sizeof(float) * 1 * 16 * 16 * 64);

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

	free(cpuInputPtr);
	free(cpuOutputPtr);
}

TEST(AvgPooling2d, backwardGPU_float) {
    typedef float real;

	GPUDevice device;

    auto cpuInputValuePtr = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);
	auto cpuInputGradPtr  = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);

	auto cpuOutputValuePtr = (real*)malloc(sizeof(real) * 1 * 16 * 16 * 64);
	auto cpuOutputGradPtr  = (real*)malloc(sizeof(real) * 1 * 16 * 16 * 64);

    auto inputValue = createTensor(device, cpuInputValuePtr, ElementType::from<real>(), 1, {32, 32, 64});
    auto inputGrad  = createTensor(device, cpuInputGradPtr, ElementType::from<real>(), 1, {32, 32, 64});

    auto outputValue = createTensor(device, cpuOutputValuePtr, ElementType::from<real>(), 1, {16, 16, 64});
    auto outputGrad = createTensor(device, cpuOutputGradPtr, ElementType::from<real>(), 1, {16, 16, 64});

    /**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>() , 1, {32, 32, 64 });

    std::vector<Node*> inputs = {&inputVar};
	AvgPooling2d vagPooling(inputs, true, 3, 3, 2, 2);

	zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

	vagPooling.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(inputGrad.raw(), cpuInputGradPtr, sizeof(float) * 1 * 32 * 32 * 64);

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
		ASSERT_TRUE(std::abs(tempinputgradptr[i] - cpuInputGradPtr[i]) < 1e-5);
	}

	free(tempinputgradptr);

    free(cpuInputValuePtr);
	free(cpuInputGradPtr);
	free(cpuOutputValuePtr);
	free(cpuOutputGradPtr);


}

#ifdef HAVE_HALF

TEST(AvgPooling2d, half_GPU) {
	typedef half real;

	GPUDevice device;

    auto inputValue  = createTensor(device, ElementType::from<real>(), 1, {32, 32, 64});
    auto inputGrad   = createTensor(device, ElementType::from<real>(), 1, {32, 32, 64});
    auto outputValue = createTensor(device, ElementType::from<real>(), 1, {16, 16, 64});
    auto outputGrad  = createTensor(device, ElementType::from<real>(), 1, {16, 16, 64});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>(), 1, {32, 32, 64 });

	std::vector<Node*> inputs = { &inputVar };
	AvgPooling2d vagPooling(inputs, true, 3, 3, 2, 2);

	std::vector<const Tensor*> inputValues = { &inputValue };

	vagPooling.forward(inputValues, &outputValue);
	vagPooling.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_AVGPOOLING2DTEST_H
