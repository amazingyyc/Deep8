#include <iostream>
#include <gtest/gtest.h>

#include "basic/Basic.h"
#include "basic/GPUBasic.h"
#include "basic/Exception.h"
#include "basic/GPUException.h"
#include "model/ElementType.h"
#include "model/GPUDevice.h"

#include "TestUtils.h"

//#include "AbsTest.h"
//#include "AddTest.h"
//#include "AvgPooling2dTest.h"
//#include "Conv2dTest.h"
//#include "DeConv2dTest.h"
//#include "DivideTest.h"
#include "DotTest.h"
//#include "ExpTest.h"
//#include "L1NormTest.h"
//#include "L2NormTest.h"
//#include "LinearTest.h"
//#include "LogTest.h"
//#include "LReLuTest.h"
//#include "MatrixMultiplyTest.h"
//#include "MaxPooling2dTest.h"
//#include "MinusTest.h"
//#include "MultiplyTest.h"
//#include "ReLuTest.h"
//#include "ReduceMeanTest.h"
//#include "ReduceSumTest.h"
//#include "SigmoidTest.h"
//#include "SoftmaxTest.h"
//#include "SquareTest.h"
//#include "TanhTest.h"

//#include "LinearRegressionTest.h"?
//#include "AutoBatchTest.h"?

#ifdef HAVE_CUDA
//#include "LinearRegressionGPUTest.h"
#endif

using namespace Deep8;

int main(int argc, char *argv[]) {

	srand((unsigned)time(NULL));

	testing::InitGoogleTest(&argc, argv);

	RUN_ALL_TESTS();

#if defined(_MSC_VER)
	getchar();
#endif

	return 0;
}