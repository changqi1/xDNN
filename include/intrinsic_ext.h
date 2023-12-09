#pragma once

#include <immintrin.h>

#define AMX_BF16_ROWS  16
#define AMX_BF16_COLS  16
#define AMX_BF16_COUNT 32

#define AMX_INT8_ROWS  32
#define AMX_INT8_COLS  32
#define AMX_INT8_COUNT 64

#define AVX3_BITS     512
#define AVX3_F32_NUM  16
#define AVX3_F16_NUM  32
#define AVX3_BF16_NUM 32
#define AVX3_I32_NUM  16
#define AVX3_I16_NUM  32
#define AVX3_I8_NUM   64

#define AVX2_BITS     256
#define AVX2_F32_NUM  8
#define AVX2_F16_NUM  16
#define AVX2_BF16_NUM 16
#define AVX2_I8_NUM   32

#define AVX_BITS     256
#define AVX_F32_NUM  4
#define AVX_F16_NUM  8
#define AVX_BF16_NUM 8
#define AVX_I8_NUM   16
#define AVX_I4_NUM   32

#define SSE_BITS      128
#define B128_I4_NUM   32
#define B128_I4x2_NUM 16

#define MMX_BITS      64
#define B64_I4_NUM    16
#define B64_I4x2_NUM  8

#define VNNI_BF16_ROWS 2
#define VNNI_INT8_ROWS 4

// Load BF16 and Convert BF16 to FP32
__m512 _mm512_loadu_pbh(void const *mem_addr);
__m512 _mm512_maskz_loadu_pbh(__mmask16 k, void const *mem_addr);

// Convert FP32 to BF16 and Store BF16
void _mm512_storeu_pbh(void *mem_addr, __m512 a);
void _mm512_mask_storeu_pbh(void *mem_addr, __mmask16 k, __m512 a);