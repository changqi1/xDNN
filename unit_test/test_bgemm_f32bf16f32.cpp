#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <memory>

#include "bgemm_f32bf16f32.h"
#include "../utils/utils.h"

#define ACCURACY 0.03f

// Check if the transpose result is the same (between the non-transpose and transpose version)
void test_xdnn_bgemm_f32bf16f32_packb(int K, int N) {
    int packed_size = xdnn_bgemm_f32bf16f32_packb_size(K, N, 16, 64);
    ALLOC(XDNN_BF16, B, K * N);
    ALLOC(XDNN_BF16, transposedB, K * N);
    ALLOC(XDNN_BF16, packedB1, packed_size);
    ALLOC(XDNN_BF16, packedB2, packed_size);
    memset(packedB1.get(), 0, packed_size * sizeof(XDNN_BF16));
    memset(packedB2.get(), 0, packed_size * sizeof(XDNN_BF16));

    test_utils::init(B.get(), K * N, -0.25f, 0.25f);
    test_utils::transpose(N, K, B.get(), N, transposedB.get());

    xdnn_bgemm_f32bf16f32_packb(false, N, K, B.get(), N, packedB1.get(), 16, 64);
    xdnn_bgemm_f32bf16f32_packb(true, N, K, transposedB.get(), K, packedB2.get(), 16, 64);

    test_utils::validate(packed_size, packedB1.get(), packedB2.get(), 0.000001);
}

void test_xdnn_bgemm_f32bf16f32_compute(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;
    int packed_size = xdnn_bgemm_f32bf16f32_packb_size(K, N, 16, 64);

    ALLOC(float, A, M * lda);
    ALLOC(XDNN_BF16, B, K * ldb);
    ALLOC(XDNN_BF16, packedB, packed_size);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), K * ldb, -0.25f, 0.25f);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);

    xdnn_bgemm_f32bf16f32_packb(false, N, K, B.get(), ldb, packedB.get(), 16, 64);
    xdnn_bgemm_f32bf16f32_compute(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), 0.0f, C.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_bgemm_f32bf16f32_compute_biasadd(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;
    int packed_size = xdnn_bgemm_f32bf16f32_packb_size(K, N, 16, 64);

    ALLOC(float, A, M * lda);
    ALLOC(XDNN_BF16, B, K * ldb);
    ALLOC(XDNN_BF16, packedB, packed_size);
    ALLOC(float, bias, N);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), K * ldb, -0.25f, 0.25f);
    test_utils::init(bias.get(), N, -1.00f, 1.00f);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);
    test_utils::add_bias(M, N, refC.get(), ldc, bias.get());

    xdnn_bgemm_f32bf16f32_packb(false, N, K, B.get(), ldb, packedB.get(), 16, 64);
    xdnn_bgemm_f32bf16f32_compute_biasadd(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), 0.0f, C.get(), ldc, bias.get());

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_bgemm_f32bf16f32_compute_residential(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;
    int packed_size = xdnn_bgemm_f32bf16f32_packb_size(K, N, 16, 64);

    ALLOC(float, A, M * lda);
    ALLOC(XDNN_BF16, B, K * ldb);
    ALLOC(XDNN_BF16, packedB, packed_size);
    ALLOC(float, res, M * ldc);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), K * ldb, -0.25f, 0.25f);
    test_utils::init(C.get(), M * ldc, -1.00f, 1.00f);
    test_utils::init(res.get(), M * ldc, -1.00f, 1.00f);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);
    for (int i = 0; i < M * ldc; ++i) {
        refC.get()[i] = refC.get()[i] + res.get()[i];
    }

    xdnn_bgemm_f32bf16f32_packb(false, N, K, B.get(), ldb, packedB.get(), 16, 64);
    xdnn_bgemm_f32bf16f32_compute_residential(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), 0, C.get(), ldc, nullptr, res.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_bgemm_f32bf16f32_compute_resext(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;
    int packed_size = xdnn_bgemm_f32bf16f32_packb_size(K, N, 16, 64);

    float gamma = 2 + 1.0f * rand() / RAND_MAX;

    ALLOC(float, A, M * lda);
    ALLOC(XDNN_BF16, B, K * ldb);
    ALLOC(XDNN_BF16, packedB, packed_size);
    ALLOC(float, bias, N);
    ALLOC(float, res, M * ldc);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), K * ldb, -0.25f, 0.25f);
    test_utils::init(bias.get(), N, -1.00f, 1.00f);
    test_utils::init(res.get(), M * ldc, -1.00f, 1.00f);

    // refC = A * B + gamma * res
    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);
    test_utils::add_bias(M, N, refC.get(), ldc, bias.get());
    for (int i = 0; i < M * ldc; ++i) {
        refC.get()[i] = refC.get()[i] + gamma * res.get()[i];
    }

    xdnn_bgemm_f32bf16f32_packb(false, N, K, B.get(), ldb, packedB.get(), 16, 64);
    xdnn_bgemm_f32bf16f32_compute_resext(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), 0, C.get(), ldc, bias.get(), gamma, res.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_bgemm_f32bf16f32(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;

    ALLOC(float, A, M * lda);
    ALLOC(XDNN_BF16, B, K * ldb);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), K * ldb, -0.25f, 0.25f);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);
    xdnn_bgemm_f32bf16f32(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, C.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 4) {
        int m = std::stoi(argv[1]);
        int n = std::stoi(argv[2]);
        int k = std::stoi(argv[3]);

        test_xdnn_bgemm_f32bf16f32_compute(m, n, k, 0, 0, 0);
        test_xdnn_bgemm_f32bf16f32_compute(m, n, k, 4, 4, 4);

        return 0;
    }

    printf("Test xdnn_bgemm_f32bf16f32_packb:\n");
    test_xdnn_bgemm_f32bf16f32_packb(2, 60);
    test_xdnn_bgemm_f32bf16f32_packb(16, 64);
    test_xdnn_bgemm_f32bf16f32_packb(16, 80);
    test_xdnn_bgemm_f32bf16f32_packb(16, 128);
    test_xdnn_bgemm_f32bf16f32_packb(16, 200);
    test_xdnn_bgemm_f32bf16f32_packb(16, 512);
    test_xdnn_bgemm_f32bf16f32_packb(16, 550);
    test_xdnn_bgemm_f32bf16f32_packb(20, 50);
    test_xdnn_bgemm_f32bf16f32_packb(32, 64);
    test_xdnn_bgemm_f32bf16f32_packb(40, 40);
    test_xdnn_bgemm_f32bf16f32_packb(64, 64);
    test_xdnn_bgemm_f32bf16f32_packb(40, 500);
    test_xdnn_bgemm_f32bf16f32_packb(64, 512);
    test_xdnn_bgemm_f32bf16f32_packb(2, 70);
    test_xdnn_bgemm_f32bf16f32_packb(18, 60);
    test_xdnn_bgemm_f32bf16f32_packb(18, 70);
    test_xdnn_bgemm_f32bf16f32_packb(34, 130);
    test_xdnn_bgemm_f32bf16f32_packb(768, 768);
    test_xdnn_bgemm_f32bf16f32_packb(1024, 1024);
    test_xdnn_bgemm_f32bf16f32_packb(300, 772);
    test_xdnn_bgemm_f32bf16f32_packb(772, 300);

    printf("Test xdnn_bgemm_f32bf16f32_compute:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_bgemm_f32bf16f32_compute(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_bgemm_f32bf16f32_compute(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_bgemm_f32bf16f32_compute_biasadd:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_bgemm_f32bf16f32_compute_biasadd(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_bgemm_f32bf16f32_compute_biasadd(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_bgemm_f32bf16f32_compute_residential:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_bgemm_f32bf16f32_compute_residential(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_bgemm_f32bf16f32_compute_residential(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_bgemm_f32bf16f32_compute_resext:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_bgemm_f32bf16f32_compute_resext(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_bgemm_f32bf16f32_compute_resext(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    // error format: No FMA support.
    // printf("Test xdnn_bgemm_f32bf16f32:\n");
    // for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
    //     test_xdnn_bgemm_f32bf16f32(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
    //     test_xdnn_bgemm_f32bf16f32(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    // }

    return 0;
}
