# Deep8
Deep8 is a dynamic Deep Learning framework like Chainer and DyNet.

## Install
Change the setting in CMakeList.txt line 4:
```
HAVE_GPU: if build for GPU
BUILD_PYTHON: if build for python
BUILD_TEST: if build the Test
```

Build C++
```python
# Set the HAVE_GPU be TRUE if have a GPU in CMakeList.txt
# Set BUILD_TEST be TRUE to build a Test
# input the follow command
# run the "deep8_test" to Test
# add the "deep8_native.so" in C++ project
mkdir build
cd build
cmake ..
make
```

Build for Python
```python
# Set the HAVE_GPU be TRUE if have a GPU in CMakeList.txt
# Set BUILD_PYTHON be TRUE
# Run below cmd in Deep8 folder
# import the "deep8" in Python project
python setup.py install
```

## Python Demo
```python
# coding=utf-8

import numpy as np
from deep8 import *

trainer = AdagradTrainer()
executor = EagerExecutor(tr=trainer)

'''
|4,  -1|   |a|   |10|
|      | * | | = |  | ====> a = 3, b = 2
|2,   1|   |b|   |8 |
'''
x = np.array([4, -1, 2, 1], dtype=np.float32)
y = np.array([10, 8], dtype=np.float32)

w = parameter(executor, [1, 2])

input  = inputParameter(executor, [1, 2, 2], x)
output = inputParameter(executor, [1, 2], y)

for i in range(2000):
    (input * w - output).l1Norm().backward()

    print i + 1, "=>", w.valueStr()

print "The w should be around [3, 2]: ", w.valueStr()
```

## C++ Demo
```C
/**
 * |4,  -1|   |a|   |10|
 * |      | * | | = |  | ====> a = 3, b = 2
 * |2,   1|   |b|   |8 |
 */
float x[4] = { 4, -1, 2, 1 };
float y[2] = { 10, 8 };

EagerExecutorF executor(new AdagradTrainerF(), DeviceType::CPU);

auto w = parameter(&executor, { 1, 2 });

auto input  = inputParameter(&executor, { 1, 2, 2 }, x);
auto output = inputParameter(&executor, { 1, 2 }, y);

for (int i = 0; i < 1000; ++i) {
    (input * w - output).l1Norm().backward();

    std::cout << i + 1 << " => " << w.valueString() << std::endl;
}

std::cout << "the result should be around: [3, 2]" << std::endl;
```
