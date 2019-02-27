#ifndef PROJECT_LINEARLEARNINGRATEITERATOR_H
#define PROJECT_LINEARLEARNINGRATEITERATOR_H

#include "basic/Basic.h"
#include "trainer/LearningRateIterator.h"

namespace Deep8 {

class LinearDecayLearningRateIterator: public LearningRateIterator {
public:
   int64_t totalSteps;

   float startLearningRate;
   float endLearningRate;

   explicit LinearDecayLearningRateIterator(int64_t total, float start = 0.01, float end = 0);

   float nextLearningRate(int64_t steps) override;
};

}

#endif