#ifndef DEEP8_ELEMENTTYPE_H
#define DEEP8_ELEMENTTYPE_H

#include "basic/Basic.h"

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
    explicit ElementType(DType d, size_t bw, std::string n) : id(d), byteWidth(bw), name(n) {
    }

public:
    /**the element type id, it should be different with different type*/
    DType id;

    /**the element type byte width*/
    size_t byteWidth;
    
    /**the element type name*/
    std::string name;

    bool operator == (const ElementType &other) const {
        return this->id == other.id; 
    }

    bool operator != (const ElementType &other) const {
        return this->id != other.id;
    }

    static ElementType unknown() {
        return ElementType(DType::UnKnown, 0, "unknown");
    }

    template <typename T>
    bool is() const {
        DEEP8_RUNTIME_ERROR("Unknow type");
    }

    /**create a ElementType from a type*/
    template <typename T>
    static ElementType from() {
        DEEP8_RUNTIME_ERROR("Unknow type");
    }

    static ElementType from(DType type) {
        switch (type) {
        case DType::UnKnown:
            return ElementType(DType::UnKnown, 0, "unknown");
            break;
        case DType::Bool:
            return ElementType(DType::Bool, sizeof(bool), "bool");
            break;
        case DType::Uint8:
            return ElementType(DType::Uint8, sizeof(uint8_t), "uint8_t");
            break;
        case DType::Int8:
            return ElementType(DType::Int8, sizeof(int8_t), "int8_t");
            break;
        case DType::Uint16:
            return ElementType(DType::Int16, sizeof(int16_t), "int16_t");
            break;
        case DType::Int16:
            return ElementType(DType::Int16, sizeof(int16_t), "int16_t");
            break;
        case DType::Uint32:
            return ElementType(DType::Uint32, sizeof(uint32_t), "uint32_t");
            break;
        case DType::Int32:
            return ElementType(DType::Int32, sizeof(int32_t), "int32_t");
            break;
        case DType::Uint64:
            return ElementType(DType::Uint64, sizeof(uint64_t), "uint64_t");
            break;
        case DType::Int64:
            return ElementType(DType::Int64, sizeof(int64_t), "int64_t");
            break;

#ifdef HAVE_HALF
        case DType::Float16:
            return ElementType(DType::Float16, sizeof(half), "float16");
            break;
#endif

        case DType::Float32:
            return ElementType(DType::Float32, sizeof(float), "float32");
            break;
        case DType::Float64:
            return ElementType(DType::Float64, sizeof(double), "float64");
            break;
        default:
            DEEP8_RUNTIME_ERROR("the dtype is error");
            break;
        }
    }
};

template <>
inline bool ElementType::is<UnKnownType>() const {
    return this->id == DType::UnKnown;
}

template <>
inline bool ElementType::is<bool>() const {
    return this->id == DType::Bool;
}

template <>
inline bool ElementType::is<uint8_t>() const {
    return this->id == DType::Uint8;
}

template <>
inline bool ElementType::is<int8_t>() const {
    return this->id == DType::Int8;
}

template <>
inline bool ElementType::is<uint16_t>() const {
    return this->id == DType::Uint16;
}

template <>
inline bool ElementType::is<int16_t>() const {
    return this->id == DType::Int16;
}

template <>
inline bool ElementType::is<uint32_t>() const {
    return this->id == DType::Uint32;
}

template <>
inline bool ElementType::is<int32_t>() const {
    return this->id == DType::Int32;
}

template <>
inline bool ElementType::is<uint64_t>() const {
    return this->id == DType::Uint64;
}

template <>
inline bool ElementType::is<int64_t>() const {
    return this->id == DType::Int64;
}

#ifdef HAVE_HALF
template <>
inline bool ElementType::is<half>() const {
    return this->id == DType::Float16;
}
#endif 

template <>
inline bool ElementType::is<float>() const {
    return this->id == DType::Float32;
}

template <>
inline bool ElementType::is<double>() const {
    return this->id == DType::Float64;
}

template <>
inline ElementType ElementType::from<UnKnownType>() {
    return ElementType(DType::UnKnown, 0, "unknown");
}

template <>
inline ElementType ElementType::from<bool>() {
    return ElementType(DType::Bool, sizeof(bool), "bool");
}

template <>
inline ElementType ElementType::from<uint8_t>() {
    return ElementType(DType::Uint8, sizeof(uint8_t), "uint8_t");
}

template <>
inline ElementType ElementType::from<int8_t>() {
    return ElementType(DType::Int8, sizeof(int8_t), "int8_t");
}

template <>
inline ElementType ElementType::from<uint16_t>() {
    return ElementType(DType::Uint16, sizeof(uint16_t), "uint16_t");
}

template <>
inline ElementType ElementType::from<int16_t>() {
    return ElementType(DType::Int16, sizeof(int16_t), "int16_t");
}

template <>
inline ElementType ElementType::from<uint32_t>() {
    return ElementType(DType::Uint32, sizeof(uint32_t), "uint32_t");
}

template <>
inline ElementType ElementType::from<int32_t>() {
    return ElementType(DType::Int32, sizeof(int32_t), "int32_t");
}

template <>
inline ElementType ElementType::from<uint64_t>() {
    return ElementType(DType::Uint64, sizeof(uint64_t), "uint64_t");
}

template <>
inline ElementType ElementType::from<int64_t>() {
    return ElementType(DType::Int64, sizeof(int64_t), "int64_t");
}

#ifdef HAVE_HALF
template <>
inline ElementType ElementType::from<half>() {
    return ElementType(DType::Float16, sizeof(half), "float16");
}
#endif

template <>
inline ElementType ElementType::from<float>() {
    return ElementType(DType::Float32, sizeof(float), "float32");
}

template <>
inline ElementType ElementType::from<double>() {
    return ElementType(DType::Float64, sizeof(double), "float64");
}

}

#endif
