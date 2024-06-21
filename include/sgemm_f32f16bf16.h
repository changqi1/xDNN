#pragma once

#include "data_types/data_types.h"

extern "C" {
// Note: only call this function when transB=false, M=1 (as designed for next token of A: softmax(Q * Kᵀ), B: V)
void small_sgemm_f32f16bf16(bool transB, int M, int N, int K, float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb, float beta, XDNN_BF16 *C, int ldc);
}