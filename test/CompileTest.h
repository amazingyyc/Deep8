#ifndef DEEP8_COMPILETEST_H
#define DEEP8_COMPILETEST_H

#include "Abs.h"
#include "Trainer.h"
#include "DefaultExecutor.h"

namespace Deep8 {

TEST(Compile, test) {
	SGDTrainer<float> *trainer = new SGDTrainer<float>();
	auto executor = new DefaultExecutor<float>(trainer, DeviceType::GPU);

	auto input  = executor->addParameter({ 400, 200 });
	auto output = executor->addParameter({ 400, 200 });

	std::vector<Node*> inputs = { input };

	Abs<float> absFunc(inputs);
	absFunc.output = output;

	absFunc.forward();
	absFunc.backward();
}

}

#endif // !DEEP8_COMPILETEST_H
