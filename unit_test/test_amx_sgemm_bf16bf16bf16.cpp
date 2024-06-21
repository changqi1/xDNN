#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <memory>

#include "amx_sgemm_bf16bf16bf16.h"
#include "../utils/utils.h"

#define ACCURACY 0.03f

// Check if the transpose result is the same (between the non-transpose and transpose version)
void test_xdnn_small_amx_sgemm_bf16bf16bf16_packb(int K, int N) {
    int packed_size = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(K, N, 16, 64);
    ALLOC(XDNN_BF16, B, K * N);
    ALLOC(XDNN_BF16, transposedB, K * N);
    ALLOC(XDNN_BF16, packedB1, packed_size);
    ALLOC(XDNN_BF16, packedB2, packed_size);
    memset(packedB1.get(), 0, packed_size * sizeof(XDNN_BF16));
    memset(packedB2.get(), 0, packed_size * sizeof(XDNN_BF16));

    test_utils::init(B.get(), K * N, -0.25f, 0.25f);
    test_utils::transpose(N, K, B.get(), N, transposedB.get());

    xdnn_small_amx_sgemm_bf16bf16bf16_packb(false, N, K, B.get(), N, packedB1.get(), packed_size);
    xdnn_small_amx_sgemm_bf16bf16bf16_packb(false, N, K, B.get(), N, packedB2.get(), packed_size);

    test_utils::validate(packed_size, packedB1.get(), packedB2.get(), 0.000001);
}

void test_xdnn_small_amx_sgemm_bf16bf16bf16_compute(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;
    int packed_size = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(N, K, 32, 32);

    ALLOC(XDNN_BF16, A, M * lda);
    ALLOC(XDNN_BF16, B, K * ldb);
    ALLOC(XDNN_BF16, packedB, packed_size);
    ALLOC(XDNN_BF16, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), K * ldb, -0.25f, 0.25f);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);

    memset(packedB.get(), 0, packed_size * sizeof(XDNN_BF16));
    xdnn_small_amx_sgemm_bf16bf16bf16_packb(false, N, K, B.get(), N, packedB.get(), packed_size);
    xdnn_small_amx_sgemm_bf16bf16bf16_compute(M, N, K, A.get(), lda, packedB.get(), C.get(), ldc);

    // test_utils::print("B", B.get(), K, N, N);
    // test_utils::print("packedB", packedB.get(), ((K+31)/32)*32, N, N);
    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);

    ALLOC(XDNN_BF16, transposedB, K * ldb);
    test_utils::transpose(N, K, B.get(), ldb, transposedB.get());
    memset(packedB.get(), 0, packed_size * sizeof(XDNN_BF16));
    xdnn_small_amx_sgemm_bf16bf16bf16_packb(true, N, K, transposedB.get(), K, packedB.get(), packed_size);
    xdnn_small_amx_sgemm_bf16bf16bf16_compute(M, N, K, A.get(), lda, packedB.get(), C.get(), ldc);

    // test_utils::print("transposedB", transposedB.get(), N, K, K);
    // test_utils::print("packedB", packedB.get(), ((K+31)/32)*32, N, N);
    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_small_amx_sgemm_bf16bf16bf16_compute_nt(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = K + padB;
    int ldc = N + padC;
    int packed_size = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(N, K, 32, 32);

    ALLOC(XDNN_BF16, A, M * lda);
    ALLOC(XDNN_BF16, B, N * ldb);
    ALLOC(XDNN_BF16, packedB, packed_size);
    ALLOC(XDNN_BF16, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), N * ldb, -0.25f, 0.25f);

    test_utils::gemm_ref(false, true, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);

    xdnn_small_amx_sgemm_bf16bf16bf16_packb(true, N, K, B.get(), ldb, packedB.get(), packed_size);
    auto pB = B.get();
    auto ppacked = packedB.get();
    xdnn_small_amx_sgemm_bf16bf16bf16_compute(M, N, K, A.get(), lda, packedB.get(), C.get(), ldc); 

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 4) {
        int m = std::stoi(argv[1]);
        int n = std::stoi(argv[2]);
        int k = std::stoi(argv[3]);

        test_xdnn_small_amx_sgemm_bf16bf16bf16_compute(m, n, k, 0, 0, 0);

        return 0;
    }

    // Test the matmul needed in Q * Kᵀ
    printf("Test xdnn_small_amx_sgemm_bf16bf16bf16_compute (B transposed):\n");
    for (int m = 1; m <= 64; m += 1) {
        for (int n = 1; n <= 64; n += 1) {
            // K = 64, 128, 256
            test_xdnn_small_amx_sgemm_bf16bf16bf16_compute_nt(m, n, 64, 0, 0, 0);
            test_xdnn_small_amx_sgemm_bf16bf16bf16_compute_nt(m, n, 128, 0, 0, 0);
            test_xdnn_small_amx_sgemm_bf16bf16bf16_compute_nt(m, n, 256, 0, 0, 0);
        }
    }

    printf("Test xdnn_small_amx_sgemm_bf16bf16bf16_packb:\n");
    test_xdnn_small_amx_sgemm_bf16bf16bf16_packb(768, 768);

    for (int m = 1; m <= 128; m += 1) {
        for (int n = 32; n <= 128; n += 32) {
            for (int k = 1; k <= 512; k += 1) {
                test_xdnn_small_amx_sgemm_bf16bf16bf16_compute(m, n, k, 0, 0, 0);
            }
        }
    }

    // numactl -N 1 -m 3 ./unit_test/test_amx_sgemm_bf16bf16bf16 33 768 768
    printf("Test xdnn_bgemm_f32bf16f32_compute:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_small_amx_sgemm_bf16bf16bf16_compute(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
    }

    return 0;
}
