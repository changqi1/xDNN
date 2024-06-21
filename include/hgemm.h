#pragma once

#include "data_types/data_types.h"

extern "C" {
// To compute sgemm: C = alpha * A * B + beta * C
void xdnn_hgemm(bool transA, bool transB, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb,
        float beta, XDNN_FP16 *C, int ldc);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
void xdnn_hgemm_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB);

// To compute sgemm: C = alpha * A * packedB + beta * C
// Note: there is no ldb, as B is packed in compact format
void xdnn_hgemm_compute(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc);

// To compute sgemm w/ bias_add: C = SILU(alpha * A * packedB + beta * C)
void xdnn_hgemm_compute_silu(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc);

// To compute sgemm w/o bias_add: C = GELU(alpha * A * packedB + beta * C)
void xdnn_hgemm_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc);

// Extended residential
// C = alpha * A * packedB + beta * C + bias + gamma * res
// ldres, residential matrix stride
void xdnn_hgemm_compute_resext(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const float *bias,
        float gamma, const XDNN_FP16 *res, int ldres);

// C = [alpha * A * packedB + beta * C] * res
// ldres, residential matrix stride
void xdnn_hgemm_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *res, int ldres);

// To compute sgemm w/ bias_add: C = alpha * A * packedB + beta * C + bias
void xdnn_hgemm_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const float *bias);

// To compute sgemm w/ bias_add: C = RELU(alpha * A * packedB + beta * C + bias)
void xdnn_hgemm_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const float *bias);

// C = alpha * A * packedB + beta * C + bias + res
// ldres, redidential matrix stride
void xdnn_hgemm_compute_residential(bool transA, int M, int N, int K,
        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const float *bias, const XDNN_FP16 *res, int ldres);

// ================================================================================
// Below is single thread small sgemm
// ================================================================================
void small_hgemm(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb, XDNN_FP16 *C, int ldc);
}