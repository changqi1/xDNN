#pragma once

#include "data_types/data_types.h"

extern "C" {
// To compute bgemm: C = alpha * A * B + beta * C
void xdnn_bgemm_f32bf16f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *B, int ldb,
        float beta, float *C, int ldc);

int xdnn_bgemm_f32bf16f32_packb_size(int N, int K, int block_rows, int block_cols);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
void xdnn_bgemm_f32bf16f32_packb(bool transB, int N, int K, const XDNN_BF16 *B, int ldb, XDNN_BF16 *packedB, int block_rows, int block_cols);

// To compute bgemm: C = alpha * A * packedB + beta * C
// Note: there is no ldb, as B is packed in compact format
void xdnn_bgemm_f32bf16f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc);

// To compute bgemm w/ bias_add: C = SILU(alpha * A * packedB + beta * C)
void xdnn_bgemm_f32bf16f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc);

// Extended residential
// C = alpha * A * packedB + beta * C + bias + gamma * res
// ldres, residential matrix stride
void xdnn_bgemm_f32bf16f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias,
        float gamma, const float *res, int ldres);

// C = [alpha * A * packedB + beta * C] * res
// ldres, residential matrix stride
void xdnn_bgemm_f32bf16f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *res, int ldres);

// To compute bgemm w/ bias_add: C = alpha * A * packedB + beta * C + bias
void xdnn_bgemm_f32bf16f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias);

// To compute bgemm w/ bias_add: C = RELU(alpha * A * packedB + beta * C + bias)
void xdnn_bgemm_f32bf16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias);

// C = alpha * A * packedB + beta * C + bias + res
// ldres, redidential matrix stride
void xdnn_bgemm_f32bf16f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres);

// ================================================================================
// Below is single thread small bgemm
// ================================================================================
void small_bgemm_f32bf16f32(int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, float *C, int ldc);
}