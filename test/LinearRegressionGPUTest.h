#ifndef DEEP8_LINEARREGRESSIONGPUTEST_H
#define DEEP8_LINEARREGRESSIONGPUTEST_H

namespace Deep8 {

#ifdef HAVE_CUDA

TEST(LinearRegression, GPU_Test) {
	/**
     * |4,  -1|   |a|   |10|
     * |      | * | | = |  | ====> a = 3, b = 2
     * |2,   1|   |b|   |8 |
     */
    float x[4] = {4, -1, 2, 1};
    float y[2] = {10, 8};

	auto *trainer  = new AdagradTrainer<float>();
    auto *executor = new DefaultExecutor<float>(trainer, DeviceType::GPU);

	auto wP = executor->addParameter({1, 2});
    Expression<float> W(executor, wP);

    auto inputP = executor->addInputParameter({1, 2, 2});
    Expression<float> input(executor, inputP);

    auto outputP = executor->addInputParameter({1, 2});
    Expression<float> output(executor, outputP);

	inputP->feed(x);
    outputP->feed(y);

	float wPtr[2];

	for (int i = 0; i < 5000; ++i) {
		auto t3 = (input * W - output).l1Norm();

        executor->backward(t3);

		((GPUDevice*)(wP->value.device))->copyFromGPUToCPU(wP->value.data(), wPtr, sizeof(float) * 2);
        std::cout << i + 1 << " => " << "[" << wPtr[0] << "," << wPtr[1] << "]" << std::endl;
	}

	std::cout << "the result should be around: [3, 2]" << std::endl;

	delete executor;
}

#endif

}

#endif //DEEP8_LINEARREGRESSIONTESRT_H
