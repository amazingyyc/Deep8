#ifndef DEEP8_LINEARREGRESSIONTEST_H
#define DEEP8_LINEARREGRESSIONTEST_H

namespace Deep8 {

TEST(LinearRegression, test) {
	/**
	 * |4,  -1|   |a|   |10|
	 * |      | * | | = |  | ====> a = 3, b = 2
	 * |2,   1|   |b|   |8 |
	 */
	float x[4] = { 4, -1, 2, 1 };
	float y[2] = { 10, 8 };

	DefaultExecutorF executor(new AdagradTrainerF(), DeviceType::CPU);

	auto wP = executor.addParameter({ 1, 2 });
	ExpressionF W(&executor, wP);

	auto inputP = executor.addInputParameter({ 1, 2, 2 }, x);
	ExpressionF input(&executor, inputP);

	auto outputP = executor.addInputParameter({ 1, 2 }, y);
	ExpressionF output(&executor, outputP);

    for (int i = 0; i < 3000; ++i) {
        (input * W - output).l1Norm().backward();

        /**print the W*/
        auto ptr = wP->value.data();
        std::cout << i + 1 << " => " << "[" << ptr[0] << "," << ptr[1] << "]" << std::endl;
    }

    std::cout << "the result should be around: [3, 2]" << std::endl;
}

}

#endif
