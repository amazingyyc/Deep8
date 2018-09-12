#ifndef DEEP8_LINEARREGRESSIONTEST_H
#define DEEP8_LINEARREGRESSIONTEST_H

#include <iostream>
#include <random>

#include "DefaultExecutor.h"
#include "Expression.h"

namespace Deep8 {

TEST(LinearRegression, test) {
    auto *graph   = new DefaultExecutor<float>();

    auto wParameter = graph->addParameter({1, 1});
    Expression<float> W(graph, wParameter);

    float *wptr = (float*) (wParameter->value.pointer);
    wptr[0] = 20.0;

    auto inputP = graph->addInputParameter({1, 1});
    Expression<float> input(graph, inputP);

    auto outputP = graph->addInputParameter({1, 1});
    Expression<float> output(graph, outputP);

    std::vector<float> x(100);
    std::vector<float> y(100);

    std::default_random_engine rng;
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (int i = 0; i < 100; ++i) {
        x[i] = 2.f;
        y[i] = x[i] * 3.f + normal(rng) * 0.33f;
    }

    for (int i = 0; i < 100; ++i) {
        inputP->feed(&x[i]);
        outputP->feed(&y[i]);

		auto t = square(W * input - output);

        graph->backward(t);

        std::cout << i << "->" << wParameter->value.scalar() << std::endl;
    }

    std::cout << "the result should be around 3.0:" << wParameter->value.scalar() << std::endl;

    delete graph;
}

}

#endif //DEEP8_LINEARREGRESSIONTESRT_H
