#ifndef DEEP8_LINEARREGRESSIONTEST_H
#define DEEP8_LINEARREGRESSIONTEST_H

#include "EagerExecutor.h"
#include "Trainer.h"
#include "Device.h"
#include "Expression.h"

namespace Deep8 {

TEST(LinearRegression, test) {
	/**
	 * |4,  -1|   |a|   |10|
	 * |      | * | | = |  | ====> a = 3, b = 2
	 * |2,   1|   |b|   |8 |
	 */
	float x[4] = { 4, -1, 2, 1 };
	float y[2] = { 10, 8 };

	EagerExecutorF executor(new AdagradTrainerF(), DeviceType::CPU);

	auto w = parameter(&executor, { 2 });

	auto input  = parameter(&executor, { 2, 2 }, false, x);
	auto output = parameter(&executor, { 2 }, false, y);

    for (int i = 0; i < 1000; ++i) {
        (input * w - output).l1Norm().backward();

        /**print the w*/
        std::cout << i + 1 << " => " << w.valueString() << std::endl;
    }

    std::cout << "the result should be around: [3, 2]" << std::endl;
}

}

#endif
