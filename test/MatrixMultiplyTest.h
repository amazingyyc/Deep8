#ifndef DEEP8_MATRIXMULTIPLYTEST_H
#define DEEP8_MATRIXMULTIPLYTEST_H

#include "MatrixMultiply.h"

namespace Deep8 {

TEST(MatrixMultiply, forwardCPU) {
	CPUDevice device;

    auto input0 = createTensor<CPUDevice, float>(device, 10, 400, 200);
    auto input1 = createTensor<CPUDevice, float>(device, 1,  200, 300);
    auto output = createTensor<CPUDevice, float>(device, 10, 400, 300);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    MatrixMultiply<float> mm(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&input0, &input1};

    mm.forwardCPU(inputTensor, &output);

    for (int b = 0; b < 10; ++b) {
        auto inputPtr0 = input0.data() + b * 400 * 200;
        auto inputPtr1 = input1.data();
        auto outputPtr = output.data() + b * 400 * 300;

        for (int m = 0; m < 400; ++m) {
            for (int n = 0; n < 300; ++n) {
                float temp = 0;

                for (int k = 0; k < 200; ++k) {
                    temp += inputPtr0[m * 200 + k] * inputPtr1[k * 300 + n];
                }

                ASSERT_EQ(temp, outputPtr[m * 300 + n]);
            }
        }
    }

    freeTensor(device, input0);
    freeTensor(device, input1);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);
    freeFakeVariable(inputVar2);

}

TEST(MatrixMultiply, forwardCPU2) {
    size_t batch = 10;
    size_t m = 300;
    size_t k = 100;
    size_t n = 1;

	CPUDevice device;

    auto input0 = createTensor<CPUDevice, float>(device, 1, m, k);
    auto input1 = createTensor<CPUDevice, float>(device, batch,  k, n);
    auto output = createTensor<CPUDevice, float>(device, batch, m, n);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    MatrixMultiply<float> mm(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&input0, &input1};

    mm.forwardCPU(inputTensor, &output);

    for (int b = 0; b < batch; ++b) {
        auto inputPtr0 = input0.data();
        auto inputPtr1 = input1.data() + b * k * n;
        auto outputPtr = output.data() + b * m * n;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float temp = 0;

                for (int l = 0; l < k; ++l) {
                    temp += inputPtr0[i * k + l] * inputPtr1[l * n + j];
                }

                ASSERT_EQ(temp, outputPtr[i * n + j]);
            }
        }
    }

    freeTensor(device, input0);
    freeTensor(device, input1);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);
    freeFakeVariable(inputVar2);

}

TEST(MatrixMultiply, backwardCPU) {
	CPUDevice device;

	auto inputValue0 = createTensor<CPUDevice, float>(device, 10, 400, 200);
	auto inputValue1 = createTensor<CPUDevice, float>(device, 1, 200, 300);

	auto inputGrad0 = createTensor<CPUDevice, float>(device, 10, 400, 200);
	auto inputGrad1 = createTensor<CPUDevice, float>(device, 1, 200, 300);

    auto outputValue = createTensor<CPUDevice, float>(device, 10, 400, 300);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 10, 400, 300);

    auto inputVar0 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar0, &inputVar1};
    MatrixMultiply<float> matrixMultiply(inputs);

    zeroTensor(device, inputGrad0);
    zeroTensor(device, inputGrad1);

    std::vector<const Tensor<float>*> inputValues = {&inputValue0, &inputValue1};

     matrixMultiply.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad0);
     matrixMultiply.backwardCPU(inputValues, &outputValue, &outputGrad, 1, &inputGrad1);

    /**
     * test the inputGrad0
     */
     for (int b = 0; b < 10; ++b) {
         auto inputGradPtr0 = inputGrad0.data() + b * 400 * 200;
         auto inputPtr1 = inputValue1.data();
         auto outputGradPtr = outputGrad.data() + b * 400 * 300;

         for (int m = 0; m < 400; ++m) {
             for (int n = 0; n < 200; ++n) {
                 float temp = 0;

                 for (int k = 0; k < 300; ++k) {
                     temp += outputGradPtr[m * 300 + k] * inputPtr1[n * 300 + k];
                 }

				 ASSERT_TRUE(std::abs(temp - inputGradPtr0[m * 200 + n]) < 1e-6);
             }
         }
     }

     /**
      * test the inputGrad1
      */
     for (int m = 0; m < 200; ++m) {
         for (int n = 0; n < 300; ++n) {
             float temp = 0;

             for (int b = 0; b < 10; ++b) {
                 auto inputPtr0     = inputValue0.data() + b * 400 * 200;
                 auto inputGradPtr1 = inputGrad1.data();
                 auto outputGradPtr = outputGrad.data() + b * 400 * 300;

                 for (int k = 0; k < 400; ++k) {
                     temp += outputGradPtr[k * 300 + n] * inputPtr0[k * 200 + m];
                 }
             }

             ASSERT_EQ(temp, inputGrad1.data()[m * 300 + n]);
         }
     }

    freeTensor(device, inputValue0);
    freeTensor(device, inputValue1);
    freeTensor(device, inputGrad0);
    freeTensor(device, inputGrad1);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar0);
    freeFakeVariable(inputVar1);

}

TEST(MatrixMultiply, backwardCPU2) {
    size_t batch = 10;
    size_t m = 300;
    size_t k = 100;
    size_t n = 1;

	CPUDevice device;

    auto inputValue0 = createTensor<CPUDevice, float>(device, 1, m, k);
    auto inputValue1 = createTensor<CPUDevice, float>(device, batch, k, n);

    auto inputGrad0 = createTensor<CPUDevice, float>(device, 1, m, k);
    auto inputGrad1 = createTensor<CPUDevice, float>(device, batch, k, n);

    auto outputValue = createTensor<CPUDevice, float>(device, batch, m, n);
    auto outputGrad  = createTensor<CPUDevice, float>(device, batch, m, n);

    auto inputVar0 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar0, &inputVar1};
    MatrixMultiply<float> matrixMultiply(inputs);

    zeroTensor(device, inputGrad0);
    zeroTensor(device, inputGrad1);

    std::vector<const Tensor<float>*> inputValues = {&inputValue0, &inputValue1};

    matrixMultiply.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad0);
    matrixMultiply.backwardCPU(inputValues, &outputValue, &outputGrad, 1, &inputGrad1);

    /**
     * test the inputGrad0
     */
    auto temp0 = (float*) device.malloc(sizeof(float) * 1 * m * k);
    device.zero(temp0, sizeof(float) * 1 * m * k);

    for (int b = 0; b < batch; ++b) {
        auto tempPtr0 = temp0;
        auto inputPtr1 = inputValue1.data() + b * k * n;
        auto outputGradPtr = outputGrad.data() + b * m * n;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                for (int l = 0; l < n; ++l) {
                    tempPtr0[i * k + j] += outputGradPtr[i * n + l] * inputPtr1[j * n + l];
                }
            }
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            ASSERT_EQ(temp0[i * k + j], inputGrad0.data()[i * k + j]);
        }
    }

    auto temp1 = (float*) device.malloc(sizeof(float) * batch * k * n);
    device.zero(temp1, sizeof(float) * batch * k * n);

    for (int b = 0; b < batch; ++b) {
        auto inputPtr0 =  inputValue0.data();
        auto tempPtr1 = temp1 + b * k * n;
        auto outputGradPtr = outputGrad.data() + b * m * n;

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int l = 0; l < m; ++l) {
                    tempPtr1[i * n + j] += inputPtr0[l * k + i] * outputGradPtr[l * n + j];
                }
            }
        }
    }

    for (int i = 0; i < batch * k * n; ++i) {
        ASSERT_EQ(temp1[i], inputGrad1.data()[i]);
    }

    freeTensor(device, inputValue0);
    freeTensor(device, inputValue1);
    freeTensor(device, inputGrad0);
    freeTensor(device, inputGrad1);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar0);
    freeFakeVariable(inputVar1);

    device.free(temp0);
    device.free(temp1);

}

#ifdef HAVE_CUDA

TEST(MatrixMultiply, GPU1_float) {
	typedef float real;

	int batch = 10;
	int m = 400;
	int k = 200;
	int n = 300;

	auto device = new GPUDevice();

	auto input0Ptr = (real*)malloc(sizeof(real) * batch * m * k);
	auto input0GradPtr = (real*)malloc(sizeof(real) * batch * m * k);
	auto input1Ptr = (real*)malloc(sizeof(real) * 1 * k * n);
	auto input1GradPtr = (real*)malloc(sizeof(real) * 1 * k * n);

	auto outputPtr = (real*)malloc(sizeof(real) * batch * m * n);
	auto outputGradPtr = (real*)malloc(sizeof(real) * batch * m * n);

	auto input0 = createTensorGPU<real>(device, input0Ptr, batch, m, k);
	auto input0Grad = createTensorGPU<real>(device, input0GradPtr, batch, m, k);

	auto input1 = createTensorGPU<real>(device, input1Ptr, 1, k, n);
	auto input1Grad = createTensorGPU<real>(device, input1GradPtr, 1, k, n);

	auto output = createTensorGPU<real>(device, outputPtr, batch, m, n);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, batch, m, n);

	auto inputVar0 = createFakeVariable<GPUDevice, real>(device);
	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	MatrixMultiply<real> matrixMultiply(inputs);

	zeroTensor(device, input0Grad);
	zeroTensor(device, input1Grad);

	std::vector<const Tensor<float>*> inputValues = { &input0, &input1 };

	matrixMultiply.forwardGPU(inputValues, &output);
	matrixMultiply.backwardGPU(inputValues, &output, &outputGrad, 0, &input0Grad);
	matrixMultiply.backwardGPU(inputValues, &output, &outputGrad, 1, &input1Grad);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * batch * m * n);
	device->copyFromGPUToCPU(input0Grad.pointer, input0GradPtr, sizeof(real) * batch * m * k);
	device->copyFromGPUToCPU(input1Grad.pointer, input1GradPtr, sizeof(real) * 1 * k * n);

	/**test output*/
	for (int b = 0; b < batch; ++b) {
		auto tempinputPtr0 = input0Ptr + b * m * k;
		auto tempinputPtr1 = input1Ptr;
		auto tempoutputPtr = outputPtr + b * m * n;

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				real temp = 0;

				for (int l = 0; l < k; ++l) {
					temp += tempinputPtr0[i * k + l] * tempinputPtr1[l * n + j];
				}

				ASSERT_EQ(temp, tempoutputPtr[i * n + j]);
			}
		}
	}

	/**
	 * test the inputGrad0
	 */
	for (int b = 0; b < 10; ++b) {
		auto tempinputGradPtr0 = input0GradPtr + b * 400 * 200;
		auto tempinputPtr1 = input1Ptr;
		auto tempoutputGradPtr = outputGradPtr + b * 400 * 300;

		for (int m = 0; m < 400; ++m) {
			for (int n = 0; n < 200; ++n) {
				float temp = 0;

				for (int k = 0; k < 300; ++k) {
					temp += tempoutputGradPtr[m * 300 + k] * tempinputPtr1[n * 300 + k];
				}

				ASSERT_TRUE(std::abs(temp - tempinputGradPtr0[m * 200 + n]) < 1e-6);
			}
		}
	}

	/**
	 * test the inputGrad1
	 */
	for (int m = 0; m < 200; ++m) {
		for (int n = 0; n < 300; ++n) {
			float temp = 0;

			for (int b = 0; b < 10; ++b) {
				auto tempinputPtr0 = input0Ptr + b * 400 * 200;
				auto tempinputGradPtr1 = input1GradPtr;
				auto tempoutputGradPtr = outputGradPtr + b * 400 * 300;

				for (int k = 0; k < 400; ++k) {
					temp += tempoutputGradPtr[k * 300 + n] * tempinputPtr0[k * 200 + m];
				}
			}

			ASSERT_EQ(temp, input1GradPtr[m * 300 + n]);
		}
	}

	free(input0Ptr);
	free(input0GradPtr);
	free(input1Ptr);
	free(input1GradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input1);
	freeTensor(device, input1Grad);
	freeTensor(device, input0);
	freeTensor(device, input0Grad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar0);
	freeFakeVariable(inputVar1);

	delete device;
}

TEST(MatrixMultiply, GPU2_float) {
	typedef float real;

	int batch = 10;
	int m = 300;
	int k = 100;
	int n = 1;

	auto device = new GPUDevice();

	auto input0Ptr = (real*)malloc(sizeof(real) * 1 * m * k);
	auto input0GradPtr = (real*)malloc(sizeof(real) * 1 * m * k);

	auto input1Ptr = (real*)malloc(sizeof(real) * batch * k * n);
	auto input1GradPtr = (real*)malloc(sizeof(real) * batch * k * n);

	auto outputPtr = (real*)malloc(sizeof(real) * batch * m * n);
	auto outputGradPtr = (real*)malloc(sizeof(real) * batch * m * n);

	auto input0 = createTensorGPU<real>(device, input0Ptr, 1, m, k);
	auto input0Grad = createTensorGPU<real>(device, input0GradPtr, 1, m, k);

	auto input1 = createTensorGPU<real>(device, input1Ptr, batch, k, n);
	auto input1Grad = createTensorGPU<real>(device, input1GradPtr, batch, k, n);

	auto output = createTensorGPU<real>(device, outputPtr, batch, m, n);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, batch, m, n);

	auto inputVar0 = createFakeVariable<GPUDevice, real>(device);
	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	MatrixMultiply<float> matrixMultiply(inputs);

	zeroTensor(device, input0Grad);
	zeroTensor(device, input1Grad);

	std::vector<const Tensor<float>*> inputValues = { &input0, &input1 };

	matrixMultiply.forwardGPU(inputValues, &output);
	matrixMultiply.backwardGPU(inputValues, &output, &outputGrad, 0, &input0Grad);
	matrixMultiply.backwardGPU(inputValues, &output, &outputGrad, 1, &input1Grad);

	device->copyFromGPUToCPU(input0Grad.pointer, input0GradPtr, sizeof(real) * 1 * m * k);
	device->copyFromGPUToCPU(input1Grad.pointer, input1GradPtr, sizeof(real) * batch * k * n);
	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * batch * m * n);

	/**test output*/
	for (int b = 0; b < batch; ++b) {
		auto tempinputPtr0 = input0Ptr;
		auto tempinputPtr1 = input1Ptr + b * k * n;
		auto tempoutputPtr = outputPtr + b * m * n;

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				real temp = 0;

				for (int l = 0; l < k; ++l) {
					temp += tempinputPtr0[i * k + l] * tempinputPtr1[l * n + j];
				}

				ASSERT_EQ(temp, tempoutputPtr[i * n + j]);
			}
		}
	}

	/**
	* test the inputGrad0
	*/
	auto temp0 = (float*)malloc(sizeof(float) * 1 * m * k);
	memset(temp0, 0, sizeof(float) * 1 * m * k);

	for (int b = 0; b < batch; ++b) {
		auto tempPtr0 = temp0;
		auto tempinputPtr1 = input1Ptr + b * k * n;
		auto tempoutputGradPtr = outputGradPtr + b * m * n;

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < k; ++j) {
				for (int l = 0; l < n; ++l) {
					tempPtr0[i * k + j] += tempoutputGradPtr[i * n + l] * tempinputPtr1[j * n + l];
				}
			}
		}
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			ASSERT_EQ(temp0[i * k + j], input0GradPtr[i * k + j]);
		}
	}

	auto temp1 = (float*)malloc(sizeof(float) * batch * k * n);
	memset(temp1, 0, sizeof(float) * batch * k * n);

	for (int b = 0; b < batch; ++b) {
		auto tempinputPtr0 = input0Ptr;
		auto tempPtr1 = temp1 + b * k * n;
		auto tempoutputGradPtr = outputGradPtr + b * m * n;

		for (int i = 0; i < k; ++i) {
			for (int j = 0; j < n; ++j) {
				for (int l = 0; l < m; ++l) {
					tempPtr1[i * n + j] += tempinputPtr0[l * k + i] * tempoutputGradPtr[l * n + j];
				}
			}
		}
	}

	for (int i = 0; i < batch * k * n; ++i) {
		ASSERT_EQ(temp1[i], input1GradPtr[i]);
	}

	free(temp0);
	free(temp1);
	free(input0Ptr);
	free(input0GradPtr);
	free(input1Ptr);
	free(input1GradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input1);
	freeTensor(device, input1Grad);
	freeTensor(device, input0);
	freeTensor(device, input0Grad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar0);
	freeFakeVariable(inputVar1);

	delete device;
}


#ifdef HAVE_HALF

TEST(MatrixMultiply, half_GPU) {
	typedef half real;

	int batch = 10;
	int m = 400;
	int k = 200;
	int n = 300;

	auto device = new GPUDevice();

	auto input0 = createTensorGPU<real>(device, batch, m, k);
	auto input0Grad = createTensorGPU<real>(device, batch, m, k);

	auto input1 = createTensorGPU<real>(device, 1, k, n);
	auto input1Grad = createTensorGPU<real>(device, 1, k, n);

	auto output = createTensorGPU<real>(device, batch, m, n);
	auto outputGrad = createTensorGPU<real>(device, batch, m, n);

	auto inputVar0 = createFakeVariable<GPUDevice, real>(device);
	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	MatrixMultiply<real> matrixMultiply(inputs);

	zeroTensor(device, input0Grad);
	zeroTensor(device, input1Grad);

	std::vector<const Tensor<real>*> inputValues = { &input0, &input1 };

	matrixMultiply.forwardGPU(inputValues, &output);
	matrixMultiply.backwardGPU(inputValues, &output, &outputGrad, 0, &input0Grad);
	matrixMultiply.backwardGPU(inputValues, &output, &outputGrad, 1, &input1Grad);

	delete device;
}

#endif // HAVE_HALF
#endif // HAVE_CUDA


}

#endif //DEEP8_MATRIXMULTIPLY_H













