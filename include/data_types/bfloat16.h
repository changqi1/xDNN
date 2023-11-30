#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include <immintrin.h>

#include "bit_convert.h"

class XDNN_BF16 {
public:
    XDNN_BF16() = default;
    XDNN_BF16(float f);

    operator float() const;
    XDNN_BF16 &operator=(float f);
    XDNN_BF16 &operator+=(const float a);

private:
    uint16_t raw_bits_;
};

static_assert(sizeof(XDNN_BF16) == 2, "XDNN_BF16 must be 2 bytes");

inline XDNN_BF16::XDNN_BF16(float f) {
    (*this) = f;
}

inline XDNN_BF16 &XDNN_BF16::operator=(float f) {
    auto iraw = bit_convert<std::array<uint16_t, 2>>(f);
    switch (std::fpclassify(f)) {
        case FP_SUBNORMAL:
        case FP_ZERO:
            raw_bits_ = iraw[1];
            raw_bits_ &= 0x8000;
            break;
        case FP_INFINITE: raw_bits_ = iraw[1]; break;
        case FP_NAN:
            raw_bits_ = iraw[1];
            raw_bits_ |= 1 << 6;
            break;
        case FP_NORMAL:
            const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
            const uint32_t int_raw = bit_convert<uint32_t>(f) + rounding_bias;
            iraw = bit_convert<std::array<uint16_t, 2>>(int_raw);
            raw_bits_ = iraw[1];
            break;
    }

    return *this;
}

inline XDNN_BF16::operator float() const {
    std::array<uint16_t, 2> iraw = {{0, raw_bits_}};
    return bit_convert<float>(iraw);
}

inline XDNN_BF16 &XDNN_BF16::operator+=(const float a) {
    (*this) = float {*this} + a;
    return *this;
}
