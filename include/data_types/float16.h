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

class XDNN_FP16 {
public:
    XDNN_FP16() = default;
    XDNN_FP16(float val);

    operator float() const;
    XDNN_FP16 &operator=(float f);
    XDNN_FP16 &operator+=(XDNN_FP16 a);

private:
    float f();
    uint16_t raw_bits_;
};

static_assert(sizeof(XDNN_FP16) == 2, "XDNN_FP16 must be 2 bytes");

inline XDNN_FP16::XDNN_FP16(float val) {
    (*this) = val;
}

inline XDNN_FP16 &XDNN_FP16::operator=(float f) {
    uint32_t i = bit_convert<uint32_t>(f);
    uint32_t s = i >> 31;
    uint32_t e = (i >> 23) & 0xFF;
    uint32_t m = i & 0x7FFFFF;

    uint32_t ss = s;
    uint32_t mm = m >> 13;
    uint32_t r = m & 0x1FFF;
    uint32_t ee = 0;
    int32_t eee = (e - 127) + 15;

    if (0 == e) {
        ee = 0;
        mm = 0;
    } else if (0xFF == e) {
        ee = 0x1F;
        if (0 != m && 0 == mm) mm = 1;
    } else if (0 < eee && eee < 0x1F) {
        ee = eee;
        if (r > (0x1000 - (mm & 1))) {
            mm++;
            if (mm == 0x400) {
                mm = 0;
                ee++;
            }
        }
    } else if (0x1F <= eee) {
        ee = 0x1F;
        mm = 0;
    } else {
        float ff = fabsf(f) + 0.5;
        uint32_t ii = bit_convert<uint32_t>(ff);
        ee = 0;
        mm = ii & 0x7FF;
    }

    this->raw_bits_ = (ss << 15) | (ee << 10) | mm;
    return *this;
}

inline XDNN_FP16 &XDNN_FP16::operator+=(XDNN_FP16 a) {
    (*this) = float(f() + a.f());
    return *this;
}

inline XDNN_FP16::operator float() const {
    uint32_t ss = raw_bits_ >> 15;
    uint32_t ee = (raw_bits_ >> 10) & 0x1F;
    uint32_t mm = raw_bits_ & 0x3FF;

    uint32_t s = ss;
    uint32_t eee = ee - 15 + 127;
    uint32_t m = mm << 13;
    uint32_t e;

    if (0 == ee) {
        if (0 == mm) {
            e = 0;
        } else {
            return (ss ? -1 : 1) * std::scalbn((float)mm, -24);
        }
    } else if (0x1F == ee) {
        e = 0xFF;
    } else {
        e = eee;
    }

    uint32_t f = (s << 31) | (e << 23) | m;

    return bit_convert<float>(f);
}

inline float XDNN_FP16::f() { return (float)(*this); }
