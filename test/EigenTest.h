#ifndef DEEP8_EIGENTEST_H
#define DEEP8_EIGENTEST_H

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Deep8 {

struct EigenTest {
    void test() {
        /**thread pool test*/
        auto threadPool = new Eigen::ThreadPool(8);
        auto device     = new Eigen::ThreadPoolDevice(threadPool, 8);

        Eigen::Barrier barrier(static_cast<unsigned int>(10));

        auto printFunc = [&barrier] (int i) {
            std::cout << i << " ";
            barrier.Notify();
        };

        for (int i = 0; i < 10; ++i) {
            device->enqueueNoNotification(printFunc, i);
        }

        barrier.Wait();
    }
};

}

#endif //DEEP8_EIGENTEST_H
