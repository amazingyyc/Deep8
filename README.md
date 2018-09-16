# Deep8
Deep8 is a head-only dynamic Deep Learning framework like Chainer and DyNet.

## A simple Demo

```C
/**
 * |4,  -1|   |a|   |10|
 * |      | * | | = |  | ====> a = 3, b = 2
 * |2,   1|   |b|   |8 |
 * 
 * a simple Linear Regression Demo to calculate the [a, b]
 */
float x[4] = {4, -1, 2, 1};
float y[2] = {10, 8};

/**create the Trainer and Executor*/
auto *trainer  = new AdagradTrainer<float>();
auto *executor = new DefaultExecutor<float>(trainer);

/**add a Parameter that should be trained*/
auto wP = executor->addParameter({1, 2});
Expression<float> W(executor, wP);

/**2 input Parameter means the x and y*/
auto inputP = executor->addInputParameter({1, 2, 2});
Expression<float> input(executor, inputP);

auto outputP = executor->addInputParameter({1, 2});
Expression<float> output(executor, outputP);

/**feed the data to Input Parameter*/
inputP->feed(x);
outputP->feed(y);

for (int i = 0; i < 500; ++i) {
    /**use the L1Norm loss*/
    auto t3 = (input * W - output).l1Norm();

    /**backward to train the W*/
    executor->backward(t3);

    /**print the W*/
    auto ptr = wP->value.data();
    std::cout << i + 1 << " => " << "[" << ptr[0] << "," << ptr[1] << "]" << std::endl;
}

std::cout << "the result should be around: [3, 2]" << std::endl;

/**clean the executor*/
delete executor;
```