#ifndef DEEP8_UTILS_H
#define DEEP8_UTILS_H

namespace Deep8 {

static int getDeviceThreadNum() {
    int threadNum = 0;

#ifdef __GNUC__
    threadNum = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
#elif defined(_MSC_VER)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    threadNum = static_cast<int>(info.dwNumberOfProcessors);
#else
    DEEP8_RUNTIME_ERROR("the compile not supported!");
#endif

    if (0 >= threadNum) {
        threadNum = 4;
    }

    return threadNum;
}

}

#endif //DEEP8_UTILS_H
