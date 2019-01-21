#ifndef DEEP8_ELEMENT_TYPE_H
#define DEEP8_ELEMENT_TYPE_H

#include "Basic.h"

namespace Deep8 {

enum class DType {
    UnKnown = 0,
    Bool    = 1,
    Uint8   = 2,
    Int8    = 3,
    Uint16  = 4,
    Int16   = 5,
    Uint32  = 6,
    Int32   = 7,
    Uint64  = 8,
    Int64   = 9,
    Float16 = 10,
    Float32 = 11,
    Float64 = 12,
};

/**define a unknow type*/
class UnKnownType {
};

class ElementType {
private:
    explicit ElementType(int i, size_t bw, std::string n) : id(i), byteWidth(bw), name(n) {
    }

public:
    /**the element type id, it should be different with different type*/
    int id;

    /**the element type byte width*/
    size_t byteWidth;
    
    /**the element type name*/
    std::string name;

    int elementId() {
        return id;
    }

    int elementSize() {
        return byteWidth;
    }

    std::string elementName() {
        return name;
    }

    bool operator == (const ElementType &other) {
        return this->id == other.id; 
    }

    bool operator != (const ElementType &other) {
        return this->id != other.id;
    }

    template <typename T>
    bool is() {
        DEEP8_RUNTIME_ERROR("Unknow type");
    }

    template <>
    bool is<UnKnownType>() {
        return this->id == DType::UnKnown;
    }

    template <>
    bool is<bool>() {
        return this->id == DType::Bool;
    }

    template <>
    bool is<uint8_t>() {
        return this->id == DType::Uint8;
    }

    template <>
    bool is<int8_t>() {
        return this->id == DType::Int8;
    }

    template <>
    bool is<uint16_t>() {
        return this->id == DType::Uint16;
    }

    template <>
    bool is<int16_t>() {
        return this->id == DType::Int16;
    }

    template <>
    bool is<uint32_t>() {
        return this->id == DType::Uint32;
    }

    template <>
    bool is<int32_t>() {
        return this->id == DType::Int32;
    }

    template <>
    bool is<uint64_t>() {
        return this->id == DType::Uint64;
    }

    template <>
    bool is<int64_t>() {
        return this->id == DType::Int64;
    }


#ifdef HAVE_HALF
    template <>
    bool is<half>() {
        return this->id == DType::Float16;
    }
#endif 

    template <>
    bool is<float>() {
        return this->id == DType::Float32;
    }

    template <>
    bool is<double>() {
        return this->id == DType::Float64;
    }

    /**create a ElementType from a type*/
    template <typename T>
    static ElementType from() {
        DEEP8_RUNTIME_ERROR("Unknow type");
    }

    template <>
    static ElementType from<UnKnownType>() {
        return ElementType(DType::UnKnown, 0, "unknown");
    }

    template <>
    static ElementType from<bool>() {
        return ElementType(DType::Bool, sizeof(bool), "bool");
    }

    template <>
    static ElementType from<uint8_t>() {
        return ElementType(DType::Uint8, sizeof(uint8_t), "uint8_t");
    }

    template <>
    static ElementType from<int8_t>() {
        return ElementType(DType::Int8, sizeof(int8_t), "int8_t");
    }

    template <>
    static ElementType from<uint16_t>() {
        return ElementType(DType::Uint16, sizeof(uint16_t), "uint16_t", );
    }

    template <>
    static ElementType from<int16_t>() {
        return ElementType(DType::Int16, sizeof(int16_t), "int16_t", );
    }

    template <>
    static ElementType from<uint32_t>() {
        return ElementType(DType::Uint32, sizeof(uint32_t), "uint32_t", );
    }

    template <>
    static ElementType from<int32_t>() {
        return ElementType(DType::Int32, sizeof(int32_t), "int32_t", );
    }

    template <>
    static ElementType from<uint64_t>() {
        return ElementType(DType::Uint64, sizeof(uint64_t), "uint64_t");
    }

    template <>
    static ElementType from<int64_t>() {
        return ElementType(DType::Int64, sizeof(int64_t), "int64_t");
    }

#ifdef HAVE_HALF
    template <>
    static ElementType from<half>() {
        return ElementType(DType::Float16, sizeof(half), "float16");
    }
#endif

    template <>
    static ElementType from<float>() {
        return ElementType(DType::Float32, sizeof(float), "float32");
    }

    template <>
    static ElementType from<double>() {
        return ElementType(DType::Float64, sizeof(double), "float64");
    }

};

}

#endif
