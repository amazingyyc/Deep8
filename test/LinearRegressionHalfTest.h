#ifndef DEEP8_LINEARREGRESSIONHALFTEST_H
#define DEEP8_LINEARREGRESSIONHALFTEST_H

#ifdef HAVE_HALF

TEST(LinearRegression, half_Test) {
	/**
     * |4,  -1|   |a|   |10|
     * |      | * | | = |  | ====> a = 3, b = 2
     * |2,   1|   |b|   |8 |
     */
    half x[4] = {4.0, -1.0, 2.0, 1.0};
    half y[2] = {10.0, 8.0};

	DefaultExecutor<half> executor(new AdagradTrainer<half>(), DeviceType::GPU);

	auto wP = executor.addParameter({1, 2});
    ExpressionF W(&executor, wP);

    auto inputP = executor.addInputParameter({1, 2, 2}, x);
    ExpressionF input(&executor, inputP);

    auto outputP = executor.addInputParameter({1, 2}, y);
    ExpressionF output(&executor, outputP);

	float wPtr[2];

	for (int i = 0; i < 5000; ++i) {
		auto t3 = (input * W - output).l1Norm();
		t3.backward();

		((GPUDevice*)(wP->value.device))->copyFromGPUToCPU(wP->value.data(), wPtr, sizeof(float) * 2);
        std::cout << i + 1 << " => " << "[" << wPtr[0] << "," << wPtr[1] << "]" << std::endl;
	}

	std::cout << "the result should be around: [3, 2]" << std::endl;
}

#endif

#endif //DEEP8_LINEARREGRESSIONHALFTEST_H
