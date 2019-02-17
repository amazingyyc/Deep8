//#ifndef PROJECT_LEARNINGRATEITERATOR_H
//#define PROJECT_LEARNINGRATEITERATOR_H
//
//#include "Basic.h"
//
//namespace Deep8 {
//
//template <typename T>
//class LearningRateIterator {
//public:
//    virtual T generateLearningRate(int64_t steps) = 0;
//};
//
//template <typename T>
//class ConstantLearningRateIterator: public LearningRateIterator<T> {
//public:
//    T learningRate;
//
//    explicit ConstantLearningRateIterator(T lr);
//
//    T generateLearningRate(int64_t steps) override;
//};
//
//template <typename T>
//class LinearDecayLearningRateIterator: public LearningRateIterator<T> {
//public:
//    int64_t totalSteps;
//
//    T startLearningRate;
//    T endLearningRate;
//
//    explicit LinearDecayLearningRateIterator(int64_t total, T start, T end);
//
//    T generateLearningRate(int64_t steps) override;
//};
//
//
//}
//
//#endif //PROJECT_LEARNINGRATEITERATOR_H
