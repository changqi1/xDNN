#pragma once

#include "data_types/data_types.h"

extern "C" {
// To compute sgemm: C = alpha * A * (XDNN_FP16 *)B + beta * C
void xdnn_sgemm_f32f16f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb,
        float beta, float *C, int ldc);

void xdnn_sgemm_f32f16f32_single_thread(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb,
        float beta, float *C, int ldc);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
void xdnn_sgemm_f32f16f32_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB);

// To compute sgemm: C = alpha * A * (XDNN_FP16 *)packedB + beta * C
// Note: there is no ldb, as B is packed in compact format
void xdnn_sgemm_f32f16f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc);

// To compute sgemm w/ bias_add: C = SILU(alpha * A * (XDNN_FP16 *)packedB + beta * C)
void xdnn_sgemm_f32f16f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc);

// To compute sgemm w/o bias_add: C = GELU(alpha * A * packedB + beta * C)
void xdnn_sgemm_f32f16f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc);

// To compute sgemm w/ bias_add: C = alpha * A * (XDNN_FP16 *)packedB + beta * C + bias
void xdnn_sgemm_f32f16f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias);

// To compute sgemm w/ bias_add: C = RELU(alpha * A * (XDNN_FP16 *)packedB + beta * C + bias)
void xdnn_sgemm_f32f16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias);

// C = alpha * A * (XDNN_FP16 *)packedB + beta * C + bias + res
// ldres, residential matrix stride
void xdnn_sgemm_f32f16f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres);

// Extended residential
// C = alpha * A * (XDNN_FP16 *)packedB + beta * C + bias + gamma * res
// ldres, residential matrix stride
void xdnn_sgemm_f32f16f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias, 
        float gamma, const float *res, int ldres);

// C = [alpha * A * (XDNN_FP16 *)packedB + beta * C] * res
// ldres, residential matrix stride
void xdnn_sgemm_f32f16f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *res, int ldres);

// ================================================================================
// Below is single thread small sgemm
// ================================================================================
void small_sgemm_f32f16f32(int M, int N, int K, const float *A, int lda, const XDNN_FP16 *B, int ldb, float *C, int ldc);
}