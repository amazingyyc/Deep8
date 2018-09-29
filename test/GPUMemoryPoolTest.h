#ifndef DEEP8_GPUMEMORYPOOLTEST_H
#define DEEP8_GPUMEMORYPOOLTEST_H

namespace Deep8 {

#ifdef HAVE_CUDA

TEST(GPUMemoryPool, poolTest) {
	GPUMemoryPool pool(0, 256);

	auto p1 = pool.malloc(6);

	pool.printInfo();

	auto p2 = pool.malloc(56);

	pool.printInfo();

	auto p3 = pool.malloc(200);

	pool.printInfo();

	pool.free(p3);

	pool.printInfo();

	auto p4 = pool.malloc(2);

	pool.printInfo();

	pool.free(p1);

	pool.printInfo();

	pool.free(p2);

	pool.printInfo();

	pool.free(p4);

	pool.printInfo();
}

#endif // HAVE_CUDA

}



#endif //DEEP8_MEMORYPOOLTEST_H