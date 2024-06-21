#pragma once

#include "data_types/data_types.h"

extern "C" {
// Note: only call this function when transB=false, M=1 (as designed for next token of A: softmax(Q * Kᵀ), B: V)
void small_sgemm_f32bf16bf16(bool transB, int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc);

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
 * Note: only call this function when transB=false, M=1
 */
void small_sgemm_f32bf16bf16_b(bool transB, int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc, int *blockIndices, int blockStride, int blockSize);
}