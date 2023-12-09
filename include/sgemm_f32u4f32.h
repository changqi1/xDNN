#pragma once

#include "data_types/data_types.h"

extern "C" {
// Symmetric Quantization per Colums
// quantizedB = int8(B) = round(B[:, n] / abs(max(B[:, n]), min(B[:, n])) * 127) = round(B[:, n] * (127 / max(abs(max(B[:, n])), abs(min(B[:, n])))))
// scaleB = B[n] = abs(max(B[:, n]), min(B[:, n])) / 127
// quantization_rate: keep intermediate range values and discard larger values and smaller values
//         │  xx  │
//         │ x  x │
//         │ x  x │
//         │ x  x │
//         │x    x│
//        *│      │*
//      ** │      │ **
void xdnn_sgemm_f32u4f32_quantize(bool transB, int N, int K, const float *B, int ldb,
        float quantization_rate, XDNN_UINT4x2 *quantizedB, int ldqb, float *scaleB, float *zeroB);

// To compute sgemm: C = alpha * A * B + beta * C
void xdnn_sgemm_f32u4f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *B, int ldb, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
void xdnn_sgemm_f32u4f32_packb(bool transB, int N, int K, const XDNN_UINT4x2 *B, int ldb, XDNN_UINT4x2 *packedB);

// To compute sgemm: C = alpha * A * packedB + beta * C
// Note: there is no ldb, as B is packed in compact format
void xdnn_sgemm_f32u4f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc);

// To compute sgemm w/ bias_add: C = SILU(alpha * A * packedB + beta * C)
void xdnn_sgemm_f32u4f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc);

// To compute sgemm w/ bias_add: C = alpha * A * packedB + beta * C + bias
void xdnn_sgemm_f32u4f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias);

// To compute sgemm w/ bias_add: C = RELU(alpha * A * packedB + beta * C + bias)
void xdnn_sgemm_f32u4f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias);

// C = alpha * A * packedB + beta * C + bias + res
// ldres, residential matrix stride
void xdnn_sgemm_f32u4f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres);

// Extended residential
// C = alpha * A * packedB + beta * C + bias + gamma * res
// ldres, residential matrix stride
void xdnn_sgemm_f32u4f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, 
        float gamma, const float *res, int ldres);

// C = [alpha * A * packedB + beta * C] * res
// ldres, residential matrix stride
void xdnn_sgemm_f32u4f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *res, int ldres);

// ================================================================================
// Below is single thread small sgemm
// ================================================================================
void small_sgemm_f32u4f32(int M, int N, int K, const float *A, int lda,
        const XDNN_UINT4x2 *B, int ldb, const float *scaleB, const float *zeroB, float *C, int ldc);
}