#ifndef DEEP8_MATHUTILS_H
#define DEEP8_MATHUTILS_H

#include <iostream>

namespace Deep8 {

inline bool isPowerOf2(size_t size) {
    return 0 == (size & (size - 1));
}

inline size_t logOf2(size_t size) {
    size_t k = 0;

    while (size > 1) {
        size >>= 1;
        k++;
    }

    return k;
}

inline size_t nextPowerOf2(size_t size) {
    if (isPowerOf2(size)) {
        return size;
    }

    size_t ret = 1;

    while (ret < size) {
        ret <<= 1;
    }

    return ret;
}

inline size_t prevPowerOf2(size_t size) {
    if (isPowerOf2(size)) {
        return size;
    }

    size_t ret = 1;

    while (ret <= size / 2) {
        ret *= 2;
    }

    return ret;
}

}

#endif //DEEP8_MATHUTILS_H
