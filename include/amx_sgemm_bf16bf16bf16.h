#pragma once

#include "data_types/data_types.h"

extern "C" {
// To compute sgemm: C = alpha * A * B + beta * C
void xdnn_amx_sgemm_bf16bf16bf16(bool transA, bool transB, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const XDNN_BF16 *B, int ldb,
        float beta, XDNN_BF16 *C, int ldc);

void xdnn_amx_sgemm_bf16bf16bf16_single_thread(bool transA, bool transB, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const XDNN_BF16 *B, int ldb,
        float beta, XDNN_BF16 *C, int ldc);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
void xdnn_amx_sgemm_bf16bf16bf16_packb(bool transB, int N, int K, const XDNN_BF16 *B, int ldb, XDNN_BF16 *packedB);

// To compute sgemm: C = alpha * A * packedB + beta * C
// Note: there is no ldb, as B is packed in compact format
void xdnn_amx_sgemm_bf16bf16bf16_compute(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const float *packedB,
        float beta, XDNN_BF16 *C, int ldc);

// To compute sgemm w/ bias_add: C = SILU(alpha * A * packedB + beta * C)
void xdnn_amx_sgemm_bf16bf16bf16_compute_silu(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const float *packedB,
        float beta, XDNN_BF16 *C, int ldc);

// To compute sgemm w/ bias_add: C = alpha * A * packedB + beta * C + bias
void xdnn_amx_sgemm_bf16bf16bf16_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const float *packedB,
        float beta, XDNN_BF16 *C, int ldc, const XDNN_BF16 *Bias);

// To compute sgemm w/ bias_add: C = RELU(alpha * A * packedB + beta * C + bias)
void xdnn_amx_sgemm_bf16bf16bf16_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const float *packedB,
        float beta, XDNN_BF16 *C, int ldc, const XDNN_BF16 *Bias);

// C = alpha * A * packedB + beta * C + bias + res
// ldres, residential matrix stride
void xdnn_amx_sgemm_bf16bf16bf16_compute_residential(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const float *packedB,
        float beta, XDNN_BF16 *C, int ldc, const XDNN_BF16 *Bias, const float *res, int ldres);

// Extended residential
// C = alpha * A * packedB + beta * C + bias + gamma * res
// ldres, residential matrix stride
void xdnn_amx_sgemm_bf16bf16bf16_compute_resext(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const float *packedB,
        float beta, XDNN_BF16 *C, int ldc, const XDNN_BF16 *Bias, 
        float gamma, const float *res, int ldres);

// C = [alpha * A * packedB + beta * C] * res
// ldres, residential matrix stride
void xdnn_amx_sgemm_bf16bf16bf16_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16 *A, int lda, const float *packedB,
        float beta, XDNN_BF16 *C, int ldc, const float *res, int ldres);

// ================================================================================
// Below is single thread small sgemm
// ================================================================================
void xdnn_small_amx_sgemm_bf16bf16bf16(int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc);

int xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(int N, int K, int block_rows, int block_cols);
void xdnn_small_amx_sgemm_bf16bf16bf16_packb(bool transB, int N, int K, const XDNN_BF16 *B, int stride, XDNN_BF16 *packedB, int size);
void xdnn_small_amx_sgemm_bf16bf16bf16_compute(int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_BF16 *packedB, XDNN_BF16 *C, int ldc);
}