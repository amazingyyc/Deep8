#ifndef PROJECT_CONSTANTLEARNINGRATEITERATOR_H
#define PROJECT_CONSTANTLEARNINGRATEITERATOR_H

#include "basic/Basic.h"
#include "trainer/LearningRateIterator.h"

namespace Deep8 {

class ConstantLearningRateIterator: public LearningRateIterator {
public:
   float learningRate;

   explicit ConstantLearningRateIterator(float lr = 0.01);

   float nextLearningRate(int64_t steps) override;
};

}


#endif