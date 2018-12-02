#ifndef DEEP8_AUTOBATCHHELPER_H
#define DEEP8_AUTOBATCHHELPER_H

#include "Basic.h"
#include "Shape.h"
#include "Function.h"

namespace Deep8 {

/**
 * a helper class to generate the auto batch code
 */
class AutoBatchCodeHelper {
private:
    static std::string underline;
    static std::string colon;
    static std::string comma;

    static std::string arrayBegin;
    static std::string arrayEnd;
    static std::string leftCurly;
    static std::string rightCurly;

    std::ostringstream oss;

public:
    /**clear the code*/
    void clear();

    /**get the auto batch hash code*/
    size_t autoBatchCode();

    void functionType(FunctionType);
    void nodeId(int id);

    void batch(size_t);
    void row(size_t);
    void col(size_t);
    void shape(Shape&);

    void inputBegin(int);
    void inputEnd(int);

    void input0Begin();
    void input0End();

    void input1Begin();
    void input1End();

    void input2Begin();
    void input2End();

    AutoBatchCodeHelper& operator << (int);
    AutoBatchCodeHelper& operator << (size_t);
    AutoBatchCodeHelper& operator << (std::string);
    AutoBatchCodeHelper& operator << (Shape&);
};

}

#endif
