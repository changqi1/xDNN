#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <iostream>

class XDNN_UINT4x2 {
public:
    XDNN_UINT4x2() = default;
    XDNN_UINT4x2(uint8_t v1, uint8_t v2);
    XDNN_UINT4x2(uint8_t v1);

    XDNN_UINT4x2 &operator=(const XDNN_UINT4x2& other);
    bool operator!=(const XDNN_UINT4x2& other) const;
    uint8_t get_v1() const;
    uint8_t get_v2() const;
    void print() const;

private:
    uint8_t raw_bits_;
};

static_assert(sizeof(XDNN_UINT4x2) == 1, "XDNN_UINT4x2 must be 1 bytes");

inline XDNN_UINT4x2::XDNN_UINT4x2(uint8_t v1, uint8_t v2) {
    // In little-endian mode, the low-order byte is stored at
    // the low address end of memory. Merge v1 and v2.
    this->raw_bits_ = (v1 & 0x0F) | ((v2 & 0x0F) << 4);
}

inline XDNN_UINT4x2::XDNN_UINT4x2(uint8_t v1) {
    this->raw_bits_ = v1 & 0x0F;
}

inline XDNN_UINT4x2& XDNN_UINT4x2::operator=(const XDNN_UINT4x2& other) {
    if (this != &other) {
        raw_bits_ = other.raw_bits_;
    }

    return *this;
}

inline bool XDNN_UINT4x2::operator!=(const XDNN_UINT4x2& other) const {
    return raw_bits_ != other.raw_bits_;
}

inline uint8_t XDNN_UINT4x2::get_v1() const {
    return raw_bits_ & 0x0F;
}

inline uint8_t XDNN_UINT4x2::get_v2() const {
    return (raw_bits_ >> 4) & 0x0F;
}

inline void XDNN_UINT4x2::print() const {
    printf("uint4x2: 0x%x %d %d\n", raw_bits_, get_v1(), get_v2());
}