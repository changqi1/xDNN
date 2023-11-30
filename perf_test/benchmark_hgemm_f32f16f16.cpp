#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <iostream>

#include "hgemm_f32f16f16.h"
#include "../utils/utils.h"

const int L3_size = 1e9; // more than L3 size(1GB) to avoid cache effects

void benchmark_xdnn_hgemm_f32f16f16_compute(int M, int N, int K, int loop = 5) {
    int batch_size = std::ceil(L3_size / (M * N + M * K + K * N));
    if (batch_size < 5)
        batch_size = 5;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    ALLOC(float, A, batch_size * M * lda);
    ALLOC(XDNN_FP16, packedB, batch_size * K * ldb);
    ALLOC(XDNN_FP16, C, batch_size * M * ldc);

    test_utils::init(A.get(), batch_size * M * lda, std::remove_pointer<decltype(A.get())>::type(1.1));
    test_utils::init(packedB.get(), batch_size * K * ldb, std::remove_pointer<decltype(packedB.get())>::type(1.1));
    test_utils::init(C.get(), batch_size * M * ldc, std::remove_pointer<decltype(C.get())>::type(1.1));

    // Assume B is packed
    for (int b = 0; b < batch_size; ++b) {
        xdnn_hgemm_f32f16f16_compute(false, M, N, K, 1.0f, &A.get()[b * M * lda], lda, &packedB.get()[b * K * ldb], 0.0f, &C.get()[b * M * ldc], ldc);
    }

    Timer t;
    for (int i = 0; i < loop; ++i) {
        for (int b = 0; b < batch_size; ++b) {
            xdnn_hgemm_f32f16f16_compute(false, M, N, K, 1.0f, &A.get()[b * M * lda], lda, &packedB.get()[b * K * ldb], 0.0f, &C.get()[b * M * ldc], ldc);
        }
    }

    float latency = t.getTime() / (batch_size) / loop;
    float gflops = 2LL * M * N * K  / latency / 1000000;
    printf("xdnn_hgemm_f32f16f16_compute, M: %d, N: %d, K: %d, latency: %f ms, perf: %.2f gflops\n", M, N, K, latency, gflops);
}

void benchmark_xdnn_hgemm_f32f16f16_compute_biasadd(int M, int N, int K, int loop = 5) {
    int batch_size = std::ceil(L3_size / (M * N + M * K + K * N));
    if (batch_size < 5)
        batch_size = 5;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    ALLOC(float, A, batch_size * M * lda);
    ALLOC(XDNN_FP16, packedB, batch_size * K * ldb);
    ALLOC(XDNN_FP16, bias, batch_size * N);
    ALLOC(XDNN_FP16, C, batch_size * M * ldc);

    test_utils::init(A.get(), batch_size * M * lda, std::remove_pointer<decltype(A.get())>::type(1.1));
    test_utils::init(packedB.get(), batch_size * K * ldb, std::remove_pointer<decltype(packedB.get())>::type(1.1));
    test_utils::init(bias.get(), batch_size * N, std::remove_pointer<decltype(bias.get())>::type(1.1));
    test_utils::init(C.get(), batch_size * M * ldc, std::remove_pointer<decltype(C.get())>::type(1.1));

    // Assume B is packed
    for (int b = 0; b < batch_size; ++b) {
        xdnn_hgemm_f32f16f16_compute_biasadd(false, M, N, K, 1.0f, &A.get()[b * M * lda], lda, &packedB.get()[b * K * ldb], 0.0f, &C.get()[b * M * ldc], ldc, &bias.get()[b * N]);
    }

    Timer t;
    for (int i = 0; i < loop; ++i) {
        for (int b = 0; b < batch_size; ++b) {
            xdnn_hgemm_f32f16f16_compute_biasadd(false, M, N, K, 1.0f, &A.get()[b * M * lda], lda, &packedB.get()[b * K * ldb], 0.0f, &C.get()[b * M * ldc], ldc, &bias.get()[b * N]);
        }
    }

    float latency = t.getTime() / (batch_size) / loop;
    float gflops = 2LL * M * N * K  / latency / 1000000;
    printf("xdnn_hgemm_f32f16f16_compute_biasadd, M: %d, N: %d, K: %d, latency: %f ms, perf: %.2f gflops\n", M, N, K, latency, gflops);
}

void benchmark_xdnn_hgemm_f32f16f16_compute_resext(int M, int N, int K, int loop = 5) {
    int batch_size = std::ceil(L3_size / ((2 * M * N + M * K + K * N) * sizeof(float)));
    if (batch_size < 5) {
        batch_size = 5;
    }

    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    const int ldres = N;
    const float gamma = 2.5;

    ALLOC(float, A, batch_size * M * lda);
    ALLOC(XDNN_FP16, packedB, batch_size * K * ldb);
    ALLOC(XDNN_FP16, bias, batch_size * N);
    ALLOC(XDNN_FP16, res, batch_size * M * ldres);
    ALLOC(XDNN_FP16, C, batch_size * M * ldc);

    test_utils::init(A.get(), batch_size * M * lda, std::remove_pointer<decltype(A.get())>::type(1.1));
    test_utils::init(packedB.get(), batch_size * K * ldb, std::remove_pointer<decltype(packedB.get())>::type(1.1));
    test_utils::init(bias.get(), batch_size * N, std::remove_pointer<decltype(bias.get())>::type(1.1));
    test_utils::init(res.get(), batch_size * M * ldres, std::remove_pointer<decltype(res.get())>::type(1.1));
    test_utils::init(C.get(), batch_size * M * ldc, std::remove_pointer<decltype(C.get())>::type(1.1));

    // Assume B is packed
    for (int b = 0; b < batch_size; ++b) {
        xdnn_hgemm_f32f16f16_compute_resext(false, M, N, K, 1.0f, &A.get()[b * M * lda], lda, &packedB.get()[b * K * ldb],
                                0.0f, &C.get()[b * M * ldc], ldc, &bias.get()[b * N], gamma, &res.get()[b * M * ldres], ldres);
    }

    Timer t;
    for (int i = 0; i < loop; ++i) {
        for (int b = 0; b < batch_size; ++b) {
            xdnn_hgemm_f32f16f16_compute_resext(false, M, N, K, 1.0f, &A.get()[b * M * lda], lda, &packedB.get()[b * K * ldb],
                                    0.0f, &C.get()[b * M * ldc], ldc, &bias.get()[b * N], gamma, &res.get()[b * M * ldres], ldres);
        }
    }

    float latency = t.getTime() / (batch_size) / loop;
    float gflops = 2LL * M * N * K  / latency / 1000000;
    printf("xdnn_hgemm_f32f16f16_compute_resext, M: %d, N: %d, K: %d, latency: %f ms, perf: %.2f gflops\n", M, N, K, latency, gflops);
}

int main(int argc, char* argv[]) {
    if (argc == 5) {
        int m = std::stoi(argv[1]);
        int n = std::stoi(argv[2]);
        int k = std::stoi(argv[3]);
        int loop = std::stoi(argv[4]);

        benchmark_xdnn_hgemm_f32f16f16_compute(m, n, k, loop);

        return 0;
    }

    for (int i = 0; i < sizeof(perf_mnk) / sizeof(perf_mnk[0]); ++i) {
        benchmark_xdnn_hgemm_f32f16f16_compute(perf_mnk[i][0], perf_mnk[i][1], perf_mnk[i][2]);
    }

    // for (int i = 0; i < sizeof(perf_mnk) / sizeof(perf_mnk[0]); ++i) {
    //     benchmark_xdnn_hgemm_f32f16f16_compute_biasadd(perf_mnk[i][0], perf_mnk[i][1], perf_mnk[i][2]);
    // }

    // for (int i = 0; i < sizeof(perf_mnk) / sizeof(perf_mnk[0]); ++i) {
    //     benchmark_xdnn_hgemm_f32f16f16_compute_resext(perf_mnk[i][0], perf_mnk[i][1], perf_mnk[i][2]);
    // }

    return 0;
}
