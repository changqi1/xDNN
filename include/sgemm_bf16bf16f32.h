#pragma once

#include "data_types/data_types.h"

extern "C" {
// Note: only call this function when transB=true, M=1 (as designed for next token of Q * Kᵀ)
void small_sgemm_bf16bf16f32(bool transB, int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_BF16 *B, int ldb, float *C, int ldc);

/**
 * This function is specially designed for paged attention
 * Matrix B is like (blockSize=4):
 *                                   |<---- ldb ----->|
 *  ________________ ________________|_h0_|___________|_h0_|___________
 * | #head*headSize | #head*headSize |    |           |                | block0
 * |________________|________________|____|___________|________________|
 * |                |                |                |                | block1
 * |________________|________________|________________|________________|
 * |                |                |                |                | block2
 * |________________|________________|________________|________________|
 * |<--------------------------- blockStride ------------------------->|
 *
 * Note: only call this function when transB=true, M=1
 */
void small_sgemm_bf16bf16f32_b(bool transB, int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_BF16 *B, int ldb, float *C, int ldc, int *blockIndices, int blockStride, int blockSize);
}