#ifndef DEEP8_MAXPOOLING2DTEST_H
#define DEEP8_MAXPOOLING2DTEST_H

#include "MaxPooling2d.h"

namespace Deep8 {

TEST(MaxPooling2d, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, 1, 32, 32, 64);
    auto output = createTensor<CPUDevice, float>(device, 1, 15, 15, 64);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device, {1, 32, 32, 64});

    std::vector<Node*> inputs = {&inputVar1};
    MaxPooling2d<float> maxPooling2d(inputs, false, 3, 3, 2, 2);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    maxPooling2d.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            int inY = i * 2;
            int inX = j * 2;

            for (int k = 0; k < 64; ++k) {
                float maxValue = input.data()[inY * 32 * 64 + inX * 64 + k];

                for (int yy = 0; yy < 3; ++yy) {
                    for (int xx = 0; xx < 3; ++xx) {
                        if (input.data()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k] > maxValue) {
                            maxValue = input.data()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k];
                        }
                    }
                }

                ASSERT_EQ(output.data()[i * 15 * 64 + j * 64 + k], maxValue);
            }
        }
    }

    freeTensor(device, input);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);

    delete device;
}

TEST(MaxPooling2d, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, float>(device, 1, 32, 32, 64);
	auto inputGrad  = createTensor<CPUDevice, float>(device, 1, 32, 32, 64);

    auto outputValue = createTensor<CPUDevice, float>(device, 1, 16, 16, 64);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 1, 16, 16, 64);

    zeroTensor(device, inputGrad);

    auto inputVar = createFakeVariable<CPUDevice, float>(device, {1, 32, 32, 64});

    std::vector<Node*> inputs = {&inputVar};
    MaxPooling2d<float> maxPooling2d(inputs, false, 2, 2, 2, 2);

    std::vector<const Tensor<float>*> inputTensor = {&inputValue};

    maxPooling2d.backwardCPU(inputTensor, &outputValue, &outputGrad, 0, &inputGrad);

	auto tempinputgradptr = (float*)malloc(sizeof(float) * 1 * 32 * 32 * 64);
	memset(tempinputgradptr, 0, sizeof(float) * 1 * 32 * 32 * 64);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int inY = 2 * i;
            int inX = 2 * j;

            for (int k = 0; k < 64; ++k) {
                float maxValue = inputValue.data()[inY * 32 * 64 + inX * 64 + k];
                int maxY = inY;
                int maxX = inX;

                for (int yy = 0; yy < 2; ++yy) {
                    for (int xx = 0; xx < 2; ++xx) {
                        if (inputValue.data()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k] > maxValue) {
                            maxValue = inputValue.data()[(inY + yy) * 32 * 64 + (inX + xx) * 64 + k];

                            maxY = inY + yy;
                            maxX = inX + xx;
                        }
                    }
                }

				tempinputgradptr[maxY * 32 * 64 + maxX * 64 + k] += outputGrad.data()[i * 16 * 64 + j * 64 + k];
            }
        }
    }

	for (int i = 0; i < 32 * 32 * 64; ++i) {
		ASSERT_EQ(inputGrad.data()[i], tempinputgradptr[i]);
	}

	free(tempinputgradptr);
    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);

    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar);

    delete device;
}


#ifdef HAVE_CUDA

TEST(MaxPooling2d, GPU_float) {
	typedef float real;

	auto device = new GPUDevice();

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

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * 1 * 15 * 15 * 64);
	device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 1 * 32 * 32 * 64);

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

	delete device;
}

#endif

}

#endif //DEEP8_MAXPOOLING2DTEST_H
