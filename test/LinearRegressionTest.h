#ifndef DEEP8_LINEARREGRESSIONTEST_H
#define DEEP8_LINEARREGRESSIONTEST_H

#include <iostream>
#include <random>

#include "DefaultExecutor.h"
#include "Trainer.h"

namespace Deep8 {

TEST(LinearRegression, test) {
    auto *trainer = new SGDTrainer<float>();
    auto *graph   = new DefaultExecutor(trainer);

    auto wParameter = graph->addParameter<float>({1, 1});
    Expression W(graph, wParameter);

    float *wptr = (float*) (wParameter->value.pointer);
    wptr[0] = 20.0;

    auto inputP = graph->addInputParameter<float>({1, 1});
    Expression input(graph, inputP);

    auto outputP = graph->addInputParameter<float>({1, 1});
    Expression output(graph, outputP);

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

        auto t1 = multiply<float>(W, input);
        auto t2 = square<float>(minus<float>(t1, output));

        graph->backward(t2);

        std::cout << i << "->" << wParameter->value.scalar() << std::endl;
    }

    std::cout << "the result should be around 3.0:" << wParameter->value.scalar() << std::endl;

    delete trainer;
    delete graph;
}

}

#endif //DEEP8_LINEARREGRESSIONTESRT_H
