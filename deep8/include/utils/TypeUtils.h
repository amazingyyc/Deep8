#ifndef DEEP8_TYPEUTILS_H
#define DEEP8_TYPEUTILS_H

#include <iostream>
#include <string>

namespace Deep8 {

template <typename T>
std::string typeStr() {
    return "T";
}

template <>
std::string typeStr<float>() {
    return "float";
}

template <>
std::string typeStr<double>() {
    return "double";
}

#ifdef HAVE_HALF
template <>
std::string typeStr<half>() {
    return "half";
}
#endif

}

#endif //PROJECT_TYPEUTILS_H
