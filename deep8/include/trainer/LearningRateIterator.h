#ifndef PROJECT_LEARNINGRATEITERATOR_H
#define PROJECT_LEARNINGRATEITERATOR_H

#include "basic/Basic.h"

namespace Deep8 {

class LearningRateIterator {
public:
   virtual float nextLearningRate(int64_t steps) = 0;
};

}

#endif //PROJECT_LEARNINGRATEITERATOR_H
