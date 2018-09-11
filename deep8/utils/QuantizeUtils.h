//#ifndef DEEP8_QUANTIZEUTILS_H
//#define DEEP8_QUANTIZEUTILS_H
//
//#include <iostream>
//#include <cmath>
//
//#include "Eigen/Eigen"
//#include "Eigen/unsupported/Eigen/CXX11/Tensor"
//#include "basic/Exception.h"
//
//namespace Deep8 {
//
///**
// * the quantize utils ref:TensorFlow
// */
//
///**
// * quantize the in to a quantize number
// * the result may out of the T range so use the int64_t to store the result
// * like the T is uint8 range is [0, 255]
// * if the in is smaller than min or bigger max, than the result will be convert to a uint8
// * so the result is int64
// * @tparam T the quantized type
// * @param in the input float
// * @param min the min float pointer
// * @param max max float pointer
// * @return the quantized result
// */
//template <typename T>
//int64_t floatToQuantized(float in, float min, float max) {
//    auto lowestQuantized = static_cast<int64_t>(std::numeric_limits<T>::min());
//
//    if (min == max) {
//        return lowestQuantized;
//    }
//
//    int bits  = sizeof(T) * 8;
//    int steps = (1 << bits) - 1;
//
//    double range = max - min;
//
//    auto quantized = static_cast<int64_t>(round(((double)in * steps - (double)min * steps) / range));
//    quantized += lowestQuantized;
//
//    return quantized;
//}
//
//template <typename T>
//T floatToQuantizedClamped(float in, float min, float max) {
//    auto quantized = floatToQuantized<T>(in, min, max);
//
//    auto lowestValue  = static_cast<int64_t>(std::numeric_limits<T>::min());
//    auto highestValue = static_cast<int64_t>(std::numeric_limits<T>::max());
//
//    quantized = std::max(lowestValue, quantized);
//    quantized = std::min(highestValue, quantized);
//
//    return static_cast<T>(quantized);
//}
//
//template <typename T>
//float quantizedToFloat(T in, float min, float max) {
//    if (std::is_same<float, T>::pointer) {
//        return in;
//    }
//
//    if (min == max) {
//        return min;
//    }
//
//    auto lowestT  = std::numeric_limits<T>::min();
//    auto highestT = std::numeric_limits<T>::max();
//
//    if (in == lowestT) {
//        return min;
//    }
//
//    if (in == highestT) {
//        return max;
//    }
//
//    int bits  = sizeof(T) * 8;
//    int steps = 1 << bits - 1;
//
//    double range = max - min;
//
//    double ret = (double)(in - lowestT) * range / steps + min;
//
//    return static_cast<float>(ret);
//}
//
//template <class T>
//float floatForOneQuantizedLevel(float range_min, float range_max) {
//    auto highest = static_cast<int64_t>(std::numeric_limits<T>::max());
//    auto lowest  = static_cast<int64_t>(std::numeric_limits<T>::min());
//
//    float level = (range_max - range_min) / (highest - lowest);
//
//    return level;
//}
//
//template <typename T1, typename T2, typename T3>
//void quantizationRangeForMultiplication(float aMin, float aMax, float bMin, float bMax, float *cMin, float *cMax) {
//    auto aLevel = floatForOneQuantizedLevel<T1>(aMin, aMax);
//    auto bLevel = floatForOneQuantizedLevel<T2>(bMin, bMax);
//
//    auto cLevel = aLevel * bLevel;
//
//    auto cHighest = static_cast<int64_t>(std::numeric_limits<T3>::max());
//    auto cLowest  = static_cast<int64_t>(std::numeric_limits<T3>::min());
//
//    *cMin = cLevel * cLowest;
//    *cMax = cLevel * cHighest;
//}
//
///**
// * @brief convert the From quantized type to To quantized type
// */
//template <typename From, typename To>
//void requantizeInNewRangeUsingEigen(Eigen::ThreadPoolDevice& device,
//                                    From *in, float inMin, float inMax,
//                                    To *out, float outMin, float outMax,
//                                    size_t length) {
//    DEEP8_RUNTIME_ERROR("for now only support int32 to uint8 or uint8 to uint8");
//}
//
//template <int shift>
//struct rightShiftOperator {
//    int64_t operator()(const int64_t &a) const {
//        return a >> shift;
//    }
//};
//
//template <>
//void requantizeInNewRangeUsingEigen<int32_t, uint8_t>(Eigen::ThreadPoolDevice *device,
//                                                      int32_t *in, float inMin, float inMax,
//                                                      uint8_t *out, float outMin, float outMax,
//                                                      size_t length) {
//    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> inVector (in,  length);
//    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> outVector(out, length);
//
//    if (outMin == outMax) {
//        outVector.device(device) = 0;
//        return;
//    }
//
//    auto inRange  = inMax  - inMin;
//    auto outRange = outMax - outMin;
//
//    int shift = 16;
//
//    auto scale   = static_cast<int64_t>(255.0 * (1 << shift) * inRange / outRange);
//    auto offset1 = static_cast<int64_t>((inMin + inMax) * 255.0 * (1 << shift) / (2.0 * outRange));
//    auto offset2 = static_cast<int64_t>(255.0 * (1 << shift) * outMin / outRange);
//    auto offset = offset1 - offset2 + (1 << (shift - 1));
//
//    auto expression = (inVector.template cast<int64_t>() * scale).unaryExpr(rightShiftOperator<32>()) + offset;
//
//    outVector.device(device) = expression.unaryExpr(rightShiftOperator<shift>()).cwiseMax(0).cwiseMin(255).template cast<uint8_t>();
//}
//
///**
// * requantize the int32 to uint8
// */
//void requantizeInt32ToUint8CPU(Eigen::ThreadPoolDevice *device,
//                               int32_t *in, float inMin, float inMax,
//                               uint8_t *out, float outMin, float outMax,
//                               size_t length) {
//    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> inTensor(in, length);
//    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> outTensor(out, length);
//
//    if (outMin == outMax) {
//        outTensor.device(device) = 0;
//        return;
//    }
//
//    auto inRange  = inMax - inMin;
//    auto outRange = outMax - outMin;
//
//    int shift = 16;
//
//    auto scale   = static_cast<int64_t>(255.0 * (1 << shift) * inRange / outRange);
//    auto offset1 = static_cast<int64_t>((inMin + inMax) * 255.0 * (1 << shift) / (2.0 * outRange));
//    auto offset2 = static_cast<int64_t>(255.0 * (1 << shift) * outMin / outRange);
//    auto offset = offset1 - offset2 + (1 << (shift - 1));
//
//    auto expression = (inTensor.template cast<int64_t>() * scale).unaryExpr(rightShiftOperator<32>()) + offset;
//
//    outTensor.device(device) = expression.unaryExpr(rightShiftOperator<shift>()).cwiseMax(0).cwiseMin(255).template cast<uint8_t>();
//}
//
///**
// * requantize uint8 to uint8
// */
//void requantizeUint8ToUint8CPU(Eigen::ThreadPoolDevice *device,
//                               uint8_t *in, float inMin, float inMax,
//                               uint8_t *out, float outMin, float outMax,
//                               size_t length) {
//    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> inTensor (in,  length);
//    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> outTensor(out, length);
//
//    if (inMin == outMin && inMax == outMax) {
//        outTensor.device(device) = inTensor;
//        return;
//    }
//
//    if (outMin == outMax) {
//        outTensor.device(device) = 0;
//        return;
//    }
//
//    int shift = 16;
//
//    auto inRange  = inMax - inMin;
//    auto outRange = outMax - outMin;
//
//    auto scale  = static_cast<int64_t>(inRange / outRange * (1 << shift));
//    auto offset = static_cast<int64_t>((inMin - outMin) / outRange * 255 * (1 << shift) + (1 << (shift - 1)));
//
//    auto expression = (inTensor.template cast<int64_t>() * scale) + offset;
//
//    outTensor.device(device) = expression.unaryExpr(rightShiftOperator<shift>()).cwiseMax(0).cwiseMin(255).template cast<uint8_t>();
//}
//
//
//void requantizeInNewRangeUsingEigen<uint8_t, uint8_t>(Eigen::ThreadPoolDevice *device,
//                                                      uint8_t *in, float inMin, float inMax,
//                                                      uint8_t *out, float outMin, float outMax,
//                                                      size_t length) {
//    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> inVector (in,  length);
//    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> outVector(out, length);
//
//    if (inMin == outMin && inMax == outMax) {
//        outVector.device(device) = inVector;
//        return;
//    }
//
//    if (outMin == outMax) {
//        outVector.device(device) = 0;
//        return;
//    }
//
//    int shift = 16;
//
//    auto inRange  = inMax - inMin;
//    auto outRange = outMax - outMax;
//
//    auto scale  = static_cast<int64_t>(inRange / outRange * (1 << shift));
//    auto offset = static_cast<int64_t>((inMin - outMin) / outRange * 255 * (1 << shift) + (1 << (shift - 1)));
//
//    auto expression = (inVector.template cast<int64_t>() * scale) + offset;
//
//    outVector.device(device) = expression.unaryExpr(rightShiftOperator<shift>()).cwiseMax(0).cwiseMin(255).template cast<uint8_t>();
//}
//
///**
// * @brief quantize a array float to uint8 using eigen
// */
//void quantizeFloatToUint8CPU(Eigen::ThreadPoolDevice *device, float *in, float minValue, float maxValue, uint8_t *out, size_t length) {
//    Eigen::TensorMap<Eigen::Tensor<float, 1>>   inTensor (in,  length);
//    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> outTensor(out, length);
//
//    if (minValue == maxValue) {
//        outTensor.device(device) = 0;
//        return;
//    }
//
//    float range = maxValue - minValue;
//    float scale = 255.0 / range;
//
//    auto expression = inTensor.template cast<float>() - minValue;
//    outTensor.device(device) = (expression * scale).cwiseMax(0).cwiseMin(255).template cast<uint8_t>();
//}
//
///**
// * @brief unquantize uint8 to float
// */
//void unQuantizeUint8ToFloatCPU(Eigen::ThreadPoolDevice *device, uint8_t *in, float minValue, float maxValue, float *out, size_t length) {
//    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> inTensor (in,  length);
//    Eigen::TensorMap<Eigen::Tensor<float, 1>>   outTensor(out, length);
//
//    if (minValue == maxValue) {
//        outTensor.device(device) = minValue;
//        return;
//    }
//
//    auto range  = maxValue - minValue;
//    float scale = range / 255.0F;
//
//    outTensor.device(device) = (inTensor.template cast<float>()) * scale + minValue;
//}
//
//}
//
//#endif //DEEP8_QUANTIZEUTILS_H
