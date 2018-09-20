#include <iostream>
#include <gtest/gtest.h>

/**hack complier*/
#define private public
#define protected public

#include "Deep8.h"

#include "TestUtils.h"

//#include "AbsTest.h"
//#include "AddTest.h"
//#include "AddScalarTest.h"
//#include "AvgPooling2dTest.h"
//#include "Conv2dTest.h"
//#include "DeConv2dTest.h"
//#include "DivideTest.h"
//#include "DivideScalarTest.h"
//#include "ExpTest.h"
//#include "L1NormTest.h"
//#include "L2NormTest.h"
//#include "LinearTest.h"
//#include "LogTest.h"
//#include "LReLuTest.h"
//#include "MatrixMultiplyTest.h"
//#include "MaxPooling2dTest.h"
//#include "MinusTest.h"
//#include "MinusScalarTest.h"
//#include "MultiplyTest.h"
//#include "MultiplyScalarTest.h"
//#include "PowTest.h"
//#include "ReLuTest.h"
//#include "SigmoidTest.h"
//#include "SoftmaxTest.h"
//#include "SquareTest.h"
//#include "SumElementsTest.h"
//#include "TanHTest.h"
//#include "LinearRegressionTest.h"
#include "LinearRegressionGPUTest.h"

using namespace Deep8;

int main(int argc,char *argv[]) {

    srand((unsigned)time(NULL));

    testing::InitGoogleTest(&argc, argv);

    RUN_ALL_TESTS();
	getchar();

    return 0;
}