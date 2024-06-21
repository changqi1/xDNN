#include <algorithm>
#include <cstring>

#include "../utils/utils.h"
#include "sgemm_bf16bf16f32.h"

#define ACCURACY 0.0001f

template <typename T>
static void gemm_ref(T *A, T *B, float *C, int K, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = 0.0f;
        for (int j = 0; j < K; ++j) {
            C[i] += (float)A[j] * (float)B[i * K + j];
        }
    }
}

static void test_small_gemm_b(const int M, const int N, const int K) {
    const int headSize = K;
    const int headNum = 16;
    const int blockSize = 4;
    const int blockStride = headSize * headNum * blockSize;

    const int lda = K;
    const int ldb = headSize * headNum;
    const int ldc = N;

    int blockIndices[] = {0, 2, 5};
    int blocks = *std::max_element(blockIndices, blockIndices + sizeof(blockIndices) / sizeof(blockIndices[0])) + 1;

    ALLOC(XDNN_BF16, A, M * K);
    ALLOC(XDNN_BF16, refB, N * K);
    ALLOC(float, refC, M * N);

    test_utils::init(A.get(), M * K, -1.0f, 1.0f);
    test_utils::init(refB.get(), N * K, -1.0f, 1.0f);

    ALLOC(XDNN_BF16, B, blockStride * blocks);
    ALLOC(float, C, M * N);

    // Copy refB to B which is like the paged cache
    XDNN_BF16 *psrc = refB.get();
    for (int i = 0; i < N / blockSize; ++i) {
        XDNN_BF16 *pB = B.get() + blockIndices[i] * blockStride;
        for (int j = 0; j < blockSize; ++j) {
            memcpy(pB + j * ldb, psrc, K * sizeof(XDNN_BF16));
            psrc += K;
        }
    }
    if (N % blockSize) { // remain
        XDNN_BF16 *pB = B.get() + blockIndices[N / blockSize] * blockStride;
        for (int j = 0; j < N % blockSize; ++j) {
            memcpy(pB + j * ldb, psrc, K * sizeof(XDNN_BF16));
            psrc += K;
        }
    }

    gemm_ref(A.get(), refB.get(), refC.get(), K, N);
    small_sgemm_bf16bf16f32_b(true, M, N, K, A.get(), lda, B.get(), ldb, C.get(), ldc, blockIndices, blockStride, blockSize);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    test_small_gemm_b(1, 9, 128);
    test_small_gemm_b(1, 10, 256);

    return 0;
}