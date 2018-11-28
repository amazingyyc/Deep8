#ifndef DEEP8_AUTOBATCH_H
#define DEEP8_AUTOBATCH_H

#include "LazyExecutor.h"
#include "Trainer.h"
#include "Device.h"
#include "Expression.h"

namespace Deep8 {

TEST(AutBatch, test) {
	float x1[2] = { 5, 6 };
	float x2[2] = { 10, 8 };

	LazyExecutorF executor(new AdagradTrainerF(), DeviceType::CPU);

	auto w = parameter(&executor, { 2, 2 });

	auto i1 = parameter(&executor, { 2 }, false, x1);
	auto i2 = parameter(&executor, { 2 }, false, x2);

	for (int i = 0; i < 100; ++i) {
		auto output = (w * i1 - w * i2).l1Norm();

		output.forward();
		output.backward();

		/**print the w*/
		std::cout << i + 1 << " => " << w.valueString() << std::endl;
	}
}

}

#endif