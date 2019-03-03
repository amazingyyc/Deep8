#ifndef DEEP8_EXCEPTION_H
#define DEEP8_EXCEPTION_H

namespace Deep8 {

#define DEEP8_ASSERT(condition, message)        \
    if (!(condition)) {                         \
        std::ostringstream oss;                 \
        oss << __FILE__ << ":";                 \
        oss << __LINE__ << ": ";                 \
        oss << message  << ".";                 \
        throw std::runtime_error(oss.str());    \
    }

#define DEEP8_RUNTIME_ERROR(message) {      \
    std::ostringstream oss;                 \
    oss << __FILE__ << ":";                 \
    oss << __LINE__ << ": ";                 \
    oss << message  << ".";                 \
    throw std::runtime_error(oss.str());    \
    }

#define DEEP8_ARGUMENT_CHECK(condition, message)        \
    if (!(condition)) {                                 \
        std::ostringstream oss;                         \
        oss << __FILE__ << ":";							\
        oss << __LINE__ << ": ";						\
        oss << message  << ".";                         \
        throw std::invalid_argument(oss.str());         \
    }

}

#endif //DEEP8_EXCEPTION_H
