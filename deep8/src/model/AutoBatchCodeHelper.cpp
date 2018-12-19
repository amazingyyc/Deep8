#include "AutoBatchCodeHelper.h"

namespace Deep8 {

std::string AutoBatchCodeHelper::underline = "_";
std::string AutoBatchCodeHelper::colon = ":";
std::string AutoBatchCodeHelper::comma = ",";

std::string AutoBatchCodeHelper::arrayBegin = "[";
std::string AutoBatchCodeHelper::arrayEnd   = "]";
std::string AutoBatchCodeHelper::leftCurly  = "{";
std::string AutoBatchCodeHelper::rightCurly = "}";

/**clear the code*/
void AutoBatchCodeHelper::clear() {
    oss.str("");
}

/**get the auto batch hash code*/
size_t AutoBatchCodeHelper::autoBatchCode() {
    return std::hash<std::string>()(oss.str());
}

void AutoBatchCodeHelper::functionType(FunctionType type) {
    oss << static_cast<int>(type);
}

void AutoBatchCodeHelper::nodeId(int64_t id) {
    oss << underline << "nodeId" << colon << id;
}

void AutoBatchCodeHelper::batch(size_t batch) {
    oss << underline << "batch" << colon << batch;
}

void AutoBatchCodeHelper::row(size_t row) {
    oss << underline << "row" << colon << row;
}

void AutoBatchCodeHelper::col(size_t col) {
    oss << underline << "col" << colon << col;
}

void AutoBatchCodeHelper::shape(Shape& shape) {
    oss << underline << "shape" << colon;
    oss << arrayBegin;

    oss << shape.batch << comma;

    for (size_t i = 0; i < shape.nDims; ++i) {
        oss << shape.dim(i) << comma;
    }

    oss << arrayEnd;
}

void AutoBatchCodeHelper::inputBegin(int index) {
    oss << underline << "input" << index << colon << leftCurly;
}

void AutoBatchCodeHelper::inputEnd(int) {
    oss << rightCurly;
}

void AutoBatchCodeHelper::input0Begin() {
    inputBegin(0);
}

void AutoBatchCodeHelper::input0End() {
    inputEnd(0);
}

void AutoBatchCodeHelper::input1Begin() {
    inputBegin(1);
}

void AutoBatchCodeHelper::input1End() {
    inputEnd(1);
}

void AutoBatchCodeHelper::input2Begin() {
    inputBegin(2);
}

void AutoBatchCodeHelper::input2End() {
    inputEnd(2);
}

void AutoBatchCodeHelper::attachBegin() {
    oss << underline << "attach" << colon << leftCurly;
}

void AutoBatchCodeHelper::attachEnd() {
    oss << rightCurly;
}

AutoBatchCodeHelper& AutoBatchCodeHelper::operator << (int i) {
    oss << i;

    return *this;
}

AutoBatchCodeHelper& AutoBatchCodeHelper::operator << (size_t i) {
    oss << i;

    return *this;
}

AutoBatchCodeHelper& AutoBatchCodeHelper::operator << (std::string str) {
    oss << str;

    return *this;
}

AutoBatchCodeHelper& AutoBatchCodeHelper::operator << (Shape& shape) {
    oss << arrayBegin;
    oss << shape.batch << comma;

    for (size_t i = 0; i < shape.nDims; ++i) {
        oss << shape.dim(i) << comma;
    }

    oss << arrayEnd;

    return *this;
}

}