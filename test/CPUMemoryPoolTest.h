#ifndef DEEP8_MEMORYPOOLTEST_H
#define DEEP8_MEMORYPOOLTEST_H

#include "CPUMemoryPool.h"

namespace Deep8 {

TEST(CPUMemoryPool, MemoryPoolTest) {
    CPUMemoryPool pool(256);

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

}



#endif //DEEP8_MEMORYPOOLTEST_H
