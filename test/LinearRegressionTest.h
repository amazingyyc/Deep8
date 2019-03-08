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

	EagerExecutor executor;
	LinearDecayLearningRateIterator learningRate(1000);
	AdamTrainer trainer(&learningRate);

	auto w = parameter(&executor, { 2 });
	w.gaussian();

	auto input  = parameter(&executor, { 2, 2 }, false);
	auto output = parameter(&executor, { 2 }, false);

	input.feed(x);
	output.feed(y);

    for (int i = 0; i < 1000; ++i) {
        (input * w - output).l1NormLoss().backward();

		trainer.train(&executor, executor.trainableParameters());

        /**print the w*/
        std::cout << i + 1 << " => " << w.valueStr() << std::endl;
    }

    std::cout << "the result should be around: [3, 2]" << std::endl;
}

}

#endif
