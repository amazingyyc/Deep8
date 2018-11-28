#ifndef DEEP8_LINEARREGRESSIONGPUTEST_H
#define DEEP8_LINEARREGRESSIONGPUTEST_H

namespace Deep8 {

TEST(LinearRegression, GPU_Test) {
	/**
     * |4,  -1|   |a|   |10|
     * |      | * | | = |  | ====> a = 3, b = 2
     * |2,   1|   |b|   |8 |
     */
    float x[4] = {4, -1, 2, 1};
    float y[2] = {10, 8};

	EagerExecutorF executor(new AdagradTrainerF(), DeviceType::GPU);

	auto w = parameter(&executor, { 2 });

	auto input = parameter(&executor, { 2, 2 }, false, x);
	auto output = parameter(&executor, { 2 }, false, y);

	for (int i = 0; i < 1000; ++i) {
		(input * w - output).l1Norm().backward();

		std::cout << i + 1 << " => " << w.valueString() << std::endl;
	}

	std::cout << "the result should be around: [3, 2]" << std::endl;
}

}

#endif //DEEP8_LINEARREGRESSIONTESRT_H
