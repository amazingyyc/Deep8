#ifndef DEEP8_LINEARREGRESSIONTEST_H
#define DEEP8_LINEARREGRESSIONTEST_H

#include <iostream>
#include <random>

#include "DefaultExecutor.h"
#include "Expression.h"
#include "TensorInit.h"

namespace Deep8 {

TEST(LinearRegression, test) {
    /**
     * |4,  -1|   |a|   |10|
     * |      | * | | = |  | ====> a = 3, b = 2
     * |2,   1|   |b|   |8 |
     */
    float x[4] = {4, -1, 2, 1};
    float y[2] = {10, 8};

    auto *trainer  = new AdagradTrainer<float>();
    auto *executor = new DefaultExecutor<float>(trainer);

    auto wP = executor->addParameter({1, 2});
    Expression<float> W(executor, wP);

    auto inputP = executor->addInputParameter({1, 2, 2});
    Expression<float> input(executor, inputP);

    auto outputP = executor->addInputParameter({1, 2});
    Expression<float> output(executor, outputP);

    /**init the parameter*/
    TensorInit::gaussian<float>(wP->value);

    for (int i = 0; i < 500; ++i) {
        inputP->feed(x);
        outputP->feed(y);

        auto t3 = l1Norm(matrixMultiply(input, W) - output);

        std::cout << "loss=>" << ((Variable<float>*)t3.node)->value.scalar() << std::endl;

        executor->backward(t3);

        auto ptr = wP->value.data();

        std::cout << i + 1 << "=>" << "[" << ptr[0] << "," << ptr[1] << "]" << std::endl;
    }

    std::cout << "the result should be around: [3, 2]" << std::endl;

    delete executor;
}

}

#endif //DEEP8_LINEARREGRESSIONTESRT_H
