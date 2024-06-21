#include <algorithm>
#include <cstring>

#include "../utils/utils.h"
#include "sgemm_f32f16bf16.h"

#define ACCURACY 0.0001f

template <typename T1, typename T2, typename T3>
static void gemm_ref(T1 *A, T2 *B, T3 *C, int K, int N, bool acc) {
    for (int i = 0; i < N; ++i) {
        float res = acc ? (float)C[i] : 0.0f;
        for (int j = 0; j < K; ++j) {
            res += (float)A[j] * (float)B[j * N + i];
        }
        C[i] = (T3)res;
    }
}

static void test_small_gemm(const int M, const int N, const int K, bool acc = false) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    ALLOC(float, A, M * K);
    ALLOC(XDNN_FP16, B, K * N);
    ALLOC(XDNN_BF16, C, M * N);
    ALLOC(XDNN_BF16, refC, M * N);

    test_utils::init(A.get(), M * K, -1.0f, 1.0f);
    test_utils::init(B.get(), N * K, -1.0f, 1.0f);
    test_utils::init(C.get(), M * N, -1.0f, 1.0f);
    memcpy(refC.get(), C.get(), M * N * sizeof(XDNN_BF16));

    gemm_ref(A.get(), B.get(), refC.get(), K, N, acc);
    small_sgemm_f32f16bf16(false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, acc ? 1.0f : 0.0f, C.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    test_small_gemm(1, 128, 1);
    test_small_gemm(1, 256, 100);
    test_small_gemm(1, 128, 200);

    test_small_gemm(1, 128, 101, true);
    test_small_gemm(1, 100, 69, true);
    test_small_gemm(1, 256, 11, true);

    return 0;
}