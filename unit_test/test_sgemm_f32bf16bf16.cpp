#include <algorithm>
#include <cstring>

#include "../utils/utils.h"
#include "sgemm_f32bf16bf16.h"

#define ACCURACY 0.0001f

template <typename T1, typename T2, typename T3>
static void gemm_ref(T1 *A, T2 *B, T3 *C, int K, int N) {
    for (int i = 0; i < N; ++i) {
        float res = 0.0f;
        for (int j = 0; j < K; ++j) {
            res += (float)A[j] * (float)B[j * N + i];
        }
        C[i] = (T3)res;
    }
}

static void test_small_gemm(const int M, const int N, const int K) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    ALLOC(float, A, M * K);
    ALLOC(XDNN_BF16, B, K * N);
    ALLOC(XDNN_BF16, C, M * N);
    ALLOC(XDNN_BF16, refC, M * N);

    test_utils::init(A.get(), M * K, -1.0f, 1.0f);
    test_utils::init(B.get(), N * K, -1.0f, 1.0f);

    gemm_ref(A.get(), B.get(), refC.get(), K, N);
    small_sgemm_f32bf16bf16(false, M, N, K, A.get(), lda, B.get(), ldb, C.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

static void test_small_gemm_b(const int M, const int N, const int K) {
    const int headSize = N;
    const int headNum = 16;
    const int blockSize = 4;
    const int blockStride = headSize * headNum * blockSize;

    const int lda = K;
    const int ldb = headSize * headNum;
    const int ldc = N;

    int blockIndices[] = {0, 2, 5};
    int blocks = *std::max_element(blockIndices, blockIndices + sizeof(blockIndices) / sizeof(blockIndices[0])) + 1;

    ALLOC(float, A, M * K);
    ALLOC(XDNN_BF16, refB, N * K);
    ALLOC(XDNN_BF16, refC, M * N);

    test_utils::init(A.get(), M * K, -1.0f, 1.0f);
    test_utils::init(refB.get(), N * K, -1.0f, 1.0f);

    ALLOC(XDNN_BF16, B, blockStride * blocks);
    ALLOC(XDNN_BF16, C, M * N);

    // Copy refB to B which is like the paged cache
    XDNN_BF16 *psrc = refB.get();
    for (int i = 0; i < K / blockSize; ++i) {
        XDNN_BF16 *pB = B.get() + blockIndices[i] * blockStride;
        for (int j = 0; j < blockSize; ++j) {
            memcpy(pB + j * ldb, psrc, N * sizeof(XDNN_BF16));
            psrc += N;
        }
    }
    if (K % blockSize) { // remain
        XDNN_BF16 *pB = B.get() + blockIndices[K / blockSize] * blockStride;
        for (int j = 0; j < K % blockSize; ++j) {
            memcpy(pB + j * ldb, psrc, N * sizeof(XDNN_BF16));
            psrc += N;
        }
    }

    gemm_ref(A.get(), refB.get(), refC.get(), K, N);
    small_sgemm_f32bf16bf16_b(false, M, N, K, A.get(), lda, B.get(), ldb, C.get(), ldc, blockIndices, blockStride, blockSize);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    test_small_gemm(1, 128, 1);
    test_small_gemm(1, 256, 100);
    test_small_gemm(1, 128, 200);

    test_small_gemm_b(1, 128, 9);
    test_small_gemm_b(1, 256, 10);

    return 0;
}