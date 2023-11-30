#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <memory>

#include "hgemm_f32i8f32.h"
#include "../utils/utils.h"

#define ACCURACY 0.05f

#define ALLOC(DATATYPE, VALUE, SIZE)  std::unique_ptr<DATATYPE, decltype(&free)> VALUE(static_cast<DATATYPE*>(aligned_alloc(64, SIZE * sizeof(DATATYPE))), &free)

// Check if the transpose result is the same (between the non-transpose and transpose version)
void test_xdnn_hgemm_f32i8f32_packb(int K, int N) {
    ALLOC(int8_t, B, K * N);
    ALLOC(int8_t, transposedB, K * N);
    ALLOC(int8_t, packedB1, K * N);
    ALLOC(int8_t, packedB2, K * N);

    test_utils::init(B.get(), K * N, -127, 127);
    test_utils::transpose(N, K, B.get(), N, transposedB.get());

    xdnn_hgemm_f32i8f32_packb(false, N, K, B.get(), N, packedB1.get());
    xdnn_hgemm_f32i8f32_packb(true, N, K, transposedB.get(), K, packedB2.get());

    for (int i = 0; i < K * N; ++i) {
        if (packedB1.get()[i] != packedB2.get()[i]) {
            printf("\tFailed: packed matrix different (K=%d, N=%d, index=%d)\n", K, N, i);
            return;
        }
    }
    printf("\tPassed: K=%d, N=%d\n", K, N);
}

void test_xdnn_hgemm_f32i8f32_quantize_compute(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;

    ALLOC(float, A, M * lda);
    ALLOC(float, B, K * ldb);
    ALLOC(int8_t, quantizedB, K * ldb);
    ALLOC(int8_t, packedB, K * N);
    ALLOC(float, scaleB, N);
    ALLOC(float, zeroB, N);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(B.get(), K * ldb, -0.25f, 0.25f);
    xdnn_hgemm_f32i8f32_quantize(false, N, K, B.get(), ldb, 0.99f, quantizedB.get(), ldb, scaleB.get(), zeroB.get());

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, 0.0f, refC.get(), ldc);

    xdnn_hgemm_f32i8f32_packb(false, N, K, quantizedB.get(), ldb, packedB.get());
    xdnn_hgemm_f32i8f32_compute(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), scaleB.get(), zeroB.get(), 0.0f, C.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_hgemm_f32i8f32_compute(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;

    ALLOC(float, A, M * lda);
    ALLOC(int8_t, quantizedB, K * ldb);
    ALLOC(int8_t, packedB, K * N);
    ALLOC(float, scaleB, N);
    ALLOC(float, zeroB, N);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.0, 3.0);
    test_utils::init(quantizedB.get(), K * ldb, -127, 127);
    test_utils::init(scaleB.get(), N, -0.002, 0.002);
    test_utils::init(zeroB.get(), N, -0.05, 0.05);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, quantizedB.get(), ldb, scaleB.get(), zeroB.get(), 0.0f, refC.get(), ldc);

    xdnn_hgemm_f32i8f32_packb(false, N, K, quantizedB.get(), ldb, packedB.get());
    xdnn_hgemm_f32i8f32_compute(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), scaleB.get(), zeroB.get(), 0.0f, C.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_hgemm_f32i8f32_compute_biasadd(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;

    ALLOC(float, A, M * lda);
    ALLOC(int8_t, quantizedB, K * ldb);
    ALLOC(int8_t, packedB, K * N);
    ALLOC(float, scaleB, N);
    ALLOC(float, zeroB, N);
    ALLOC(float, bias, N);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.0, 3.0);
    test_utils::init(quantizedB.get(), K * ldb, -127, 127);
    test_utils::init(scaleB.get(), N, -0.002, 0.002);
    test_utils::init(zeroB.get(), N, -0.05, 0.05);
    test_utils::init(bias.get(), N, -0.25, 0.25);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, quantizedB.get(), ldb, scaleB.get(), zeroB.get(), 0.0f, refC.get(), ldc);
    test_utils::add_bias(M, N, refC.get(), ldc, bias.get());

    xdnn_hgemm_f32i8f32_packb(false, N, K, quantizedB.get(), ldb, packedB.get());
    xdnn_hgemm_f32i8f32_compute_biasadd(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), scaleB.get(), zeroB.get(), 0.0f, C.get(), ldc, bias.get());

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_hgemm_f32i8f32_compute_residential(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;

    ALLOC(float, A, M * lda);
    ALLOC(int8_t, quantizedB, K * ldb);
    ALLOC(int8_t, packedB, K * N);
    ALLOC(float, scaleB, N);
    ALLOC(float, zeroB, N);
    ALLOC(float, bias, N);
    ALLOC(float, res, M * ldc);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.00f, 3.00f);
    test_utils::init(quantizedB.get(), K * ldb, -127, 127);
    test_utils::init(scaleB.get(), N, -0.002, 0.002);
    test_utils::init(zeroB.get(), N, -0.05, 0.05);
    test_utils::init(bias.get(), N, -0.25, 0.25);
    test_utils::init(res.get(), M * ldc, -1.00f, 1.00f);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, quantizedB.get(), ldb, scaleB.get(), zeroB.get(), 0.0f, refC.get(), ldc);
    for (int i = 0; i < M * ldc; ++i) {
        refC.get()[i] = refC.get()[i] + res.get()[i];
    }

    xdnn_hgemm_f32i8f32_packb(false, N, K, quantizedB.get(), ldb, packedB.get());
    xdnn_hgemm_f32i8f32_compute_residential(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), scaleB.get(), zeroB.get(), 0.0f, C.get(), ldc, bias.get(), res.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_hgemm_f32i8f32_compute_resext(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;

    float gamma = 2 + 1.0f * rand() / RAND_MAX;

    ALLOC(float, A, M * lda);
    ALLOC(int8_t, quantizedB, K * ldb);
    ALLOC(int8_t, packedB, K * N);
    ALLOC(float, scaleB, N);
    ALLOC(float, zeroB, N);
    ALLOC(float, bias, N);
    ALLOC(float, res, M * ldc);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.0, 3.0);
    test_utils::init(quantizedB.get(), K * ldb, -127, 127);
    test_utils::init(scaleB.get(), N, -0.002, 0.002);
    test_utils::init(zeroB.get(), N, -0.05, 0.05);
    test_utils::init(bias.get(), N, -0.25, 0.25);
    test_utils::init(res.get(), M * ldc, -0.25, 0.25);
    test_utils::init(C.get(), M * ldc, -0.25, 0.25);
    memcpy(refC.get(), C.get(), M * ldc * sizeof(float));

    // refC = A * B + gamma * res
    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, quantizedB.get(), ldb, scaleB.get(), zeroB.get(), 0.0f, refC.get(), ldc);
    test_utils::add_bias(M, N, refC.get(), ldc, bias.get());
    for (int i = 0; i < M * ldc; ++i) {
        refC.get()[i] = refC.get()[i] + gamma * res.get()[i];
    }

    xdnn_hgemm_f32i8f32_packb(false, N, K, quantizedB.get(), ldb, packedB.get());
    xdnn_hgemm_f32i8f32_compute_resext(false, M, N, K, 1.0f, A.get(), lda, packedB.get(), scaleB.get(), zeroB.get(), 0.0f, C.get(), ldc, bias.get(), gamma, res.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

void test_xdnn_hgemm_f32i8f32(int M, int N, int K, unsigned int padA = 0, unsigned int padB = 0, unsigned int padC = 0) {
    int lda = K + padA;
    int ldb = N + padB;
    int ldc = N + padC;

    ALLOC(float, A, M * lda);
    ALLOC(int8_t, B, K * ldb);
    ALLOC(float, scaleB, N);
    ALLOC(float, zeroB, N);
    ALLOC(float, C, M * ldc);
    ALLOC(float, refC, M * ldc);

    test_utils::init(A.get(), M * lda, -3.0, 3.0);
    test_utils::init(B.get(), K * ldb, -127, 127);
    test_utils::init(scaleB.get(), N, -0.002, 0.002);
    test_utils::init(zeroB.get(), N, -0.05, 0.05);

    test_utils::gemm_ref(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, scaleB.get(), zeroB.get(), 0.0f, refC.get(), ldc);
    xdnn_hgemm_f32i8f32(false, false, M, N, K, 1.0f, A.get(), lda, B.get(), ldb, scaleB.get(), zeroB.get(), 0.0f, C.get(), ldc);

    test_utils::validate(M, N, K, lda, ldb, ldc, refC.get(), C.get(), ACCURACY);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 4) {
        int m = std::stoi(argv[1]);
        int n = std::stoi(argv[2]);
        int k = std::stoi(argv[3]);

        test_xdnn_hgemm_f32i8f32_quantize_compute(m, n, k, 0, 0, 0);
        test_xdnn_hgemm_f32i8f32_quantize_compute(m, n, k, 4, 4, 4);

        return 0;
    }

    printf("Test xdnn_hgemm_f32i8f32_packb:\n");
    test_xdnn_hgemm_f32i8f32_packb(768, 768);
    test_xdnn_hgemm_f32i8f32_packb(1024, 1024);
    test_xdnn_hgemm_f32i8f32_packb(300, 772);
    test_xdnn_hgemm_f32i8f32_packb(772, 300);

    printf("Test xdnn_hgemm_f32i8f32_quantize_compute:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_hgemm_f32i8f32_quantize_compute(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_hgemm_f32i8f32_quantize_compute(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_hgemm_f32i8f32_compute:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_hgemm_f32i8f32_compute(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_hgemm_f32i8f32_compute(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_hgemm_f32i8f32_compute_biasadd:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_hgemm_f32i8f32_compute_biasadd(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_hgemm_f32i8f32_compute_biasadd(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_hgemm_f32i8f32_compute_residential:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_hgemm_f32i8f32_compute_residential(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_hgemm_f32i8f32_compute_residential(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_hgemm_f32i8f32_compute_resext:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_hgemm_f32i8f32_compute_resext(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_hgemm_f32i8f32_compute_resext(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    printf("Test xdnn_hgemm_f32i8f32:\n");
    for (int i = 0; i < sizeof(unit_mnk) / sizeof(unit_mnk[0]); ++i) {
        test_xdnn_hgemm_f32i8f32(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 0, 0, 0);
        test_xdnn_hgemm_f32i8f32(unit_mnk[i][0], unit_mnk[i][1], unit_mnk[i][2], 4, 4, 4);
    }

    return 0;
}
