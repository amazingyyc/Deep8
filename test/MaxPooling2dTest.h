#ifndef DEEP8_MAXPOOLING2DTEST_H
#define DEEP8_MAXPOOLING2DTEST_H

#include "nodes/MaxPooling2d.h"

namespace Deep8 {

TEST(MaxPooling2d, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), 1, {32, 32, 64});
    auto output = createTensor(device, ElementType::from<float>(), 1, {15, 15, 64});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>(), 1, {32, 32, 64});

    std::vector<Node*> inputs = {&inputVar1};
    MaxPooling2d maxPooling2d(inputs, false, 3, 3, 2, 2);

    std::vector<const Tensor*> inputTensor = {&input};

    maxPooling2d.forward(inputTensor, &output);

    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            int inY = i * 2;
            int inX = j * 2;

            for (int k = 0; k < 64; ++k) {
                float maxValue = input.data<float>()[inY * 32 * 64 + inX * 64 + k];

                for (int yy = 0; yy < 3; ++yy) {
                    for (int xx = 0; xx < 3; ++xx) {
                        if (input.data<float>()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k] > maxValue) {
                            maxValue = input.data<float>()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k];
                        }
                    }
                }

                ASSERT_EQ(output.data<float>()[i * 15 * 64 + j * 64 + k], maxValue);
            }
        }
    }

}

TEST(MaxPooling2d, backwardCPU) {
	CPUDevice device;

	auto inputValue  = createTensor(device, ElementType::from<float>(), 1, {32, 32, 64});
	auto inputGrad   = createTensor(device, ElementType::from<float>(), 1, {32, 32, 64});
    auto outputValue = createTensor(device, ElementType::from<float>(), 1, {16, 16, 64});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 1, {16, 16, 64});

    zeroTensor(device, inputGrad);

    auto inputVar = createFakeVariable(device, ElementType::from<float>(), 1, {32, 32, 64});

    std::vector<Node*> inputs = {&inputVar};
    MaxPooling2d maxPooling2d(inputs, false, 2, 2, 2, 2);

    std::vector<const Tensor*> inputTensor = {&inputValue};

    maxPooling2d.backward(inputTensor, &outputValue, &outputGrad, 0, &inputGrad);

	auto tempinputgradptr = (float*)malloc(sizeof(float) * 1 * 32 * 32 * 64);
	memset(tempinputgradptr, 0, sizeof(float) * 1 * 32 * 32 * 64);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int inY = 2 * i;
            int inX = 2 * j;

            for (int k = 0; k < 64; ++k) {
                float maxValue = inputValue.data<float>()[inY * 32 * 64 + inX * 64 + k];
                int maxY = inY;
                int maxX = inX;

                for (int yy = 0; yy < 2; ++yy) {
                    for (int xx = 0; xx < 2; ++xx) {
                        if (inputValue.data<float>()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k] > maxValue) {
                            maxValue = inputValue.data<float>()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k];

                            maxY = inY + yy;
                            maxX = inX + xx;
                        }
                    }
                }

				tempinputgradptr[maxY * 32 * 64 + maxX * 64 + k] += outputGrad.data<float>()[i * 16 * 64 + j * 64 + k];
            }
        }
    }

	for (int i = 0; i < 32 * 32 * 64; ++i) {
		ASSERT_EQ(inputGrad.data<float>()[i], tempinputgradptr[i]);
	}

	free(tempinputgradptr);

}


#ifdef HAVE_CUDA

TEST(MaxPooling2d, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);

	auto outputPtr = (real*)malloc(sizeof(real) * 1 * 15 * 15 * 64);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 1 * 15 * 15 * 64);

	auto input = createTensorGPU<real>(device, inputPtr, 1, 32, 32, 64);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 1, 32, 32, 64);

	auto output = createTensorGPU<real>(device, outputPtr, 1, 15, 15, 64);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 1, 15, 15, 64);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device, { 1, 32, 32, 64 });

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar1 };
	MaxPooling2d<real> maxPooling2d(inputs, false, 3, 3, 2, 2);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	maxPooling2d.forwardGPU(inputTensor, &output);
	maxPooling2d.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 1 * 15 * 15 * 64);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 1 * 32 * 32 * 64);

	for (int i = 0; i < 15; ++i) {
		for (int j = 0; j < 15; ++j) {
			int inY = i * 2;
			int inX = j * 2;

			for (int k = 0; k < 64; ++k) {
				float maxValue = inputPtr[inY * 32 * 64 + inX * 64 + k];

				for (int yy = 0; yy < 3; ++yy) {
					for (int xx = 0; xx < 3; ++xx) {
						if (inY + yy < 32 && inX + xx < 32) {
							if (inputPtr[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k] > maxValue) {
								maxValue = inputPtr[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k];
							}
						}
					}
				}

				ASSERT_EQ(outputPtr[i * 15 * 64 + j * 64 + k], maxValue);
			}
		}
	}

	auto tempinputgradptr = (real*)malloc(sizeof(real) * 1 * 32 * 32 * 64);
	memset(tempinputgradptr, 0, sizeof(real) * 1 * 32 * 32 * 64);

	for (int i = 0; i < 15; ++i) {
		for (int j = 0; j < 15; ++j) {
			int inY = 2 * i;
			int inX = 2 * j;

			for (int k = 0; k < 64; ++k) {
				float maxValue = inputPtr[inY * 32 * 64 + inX * 64 + k];
				int maxY = inY;
				int maxX = inX;

				for (int yy = 0; yy < 3; ++yy) {
					for (int xx = 0; xx < 3; ++xx) {
						if (inputPtr[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k] > maxValue) {
							maxValue = inputPtr[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k];

							maxY = inY + yy;
							maxX = inX + xx;
						}
					}
				}

				tempinputgradptr[maxY * 32 * 64 + maxX * 64 + k] += outputGradPtr[i * 15 * 64 + j * 64 + k];
			}
		}
	}

	for (int i = 0; i < 32 * 32 * 64; ++i) {
		ASSERT_EQ(tempinputgradptr[i], inputGradPtr[i]);
	}

	free(tempinputgradptr);

	free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input);
	freeTensor(device, inputGrad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);
}


#ifdef HAVE_HALF

TEST(MaxPooling2d, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input = createTensorGPU<real>(device, 1, 32, 32, 64);
	auto inputGrad = createTensorGPU<real>(device, 1, 32, 32, 64);

	auto output = createTensorGPU<real>(device, 1, 15, 15, 64);
	auto outputGrad = createTensorGPU<real>(device, 1, 15, 15, 64);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device, { 1, 32, 32, 64 });

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar1 };
	MaxPooling2d<real> maxPooling2d(inputs, false, 3, 3, 2, 2);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	maxPooling2d.forwardGPU(inputTensor, &output);
	maxPooling2d.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_MAXPOOLING2DTEST_H
