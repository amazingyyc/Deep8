#ifndef PROJECT_LEARNINGRATEITERATOR_H
#define PROJECT_LEARNINGRATEITERATOR_H

#include "basic/Basic.h"
#include "basic/Exception.h"

namespace Deep8 {

class LearningRateIterator {
public:
   virtual float next(int64_t steps);
};

}

#endif //PROJECT_LEARNINGRATEITERATOR_H
