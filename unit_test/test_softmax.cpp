
#include "softmax.h"

static bool test_softmax_f32(int N, float scale) {
    float value = 1.0f;
    float data[N];
    for (int i = 0; i < N; ++i) {
        data[i] = value;
    }
    data[0] = value + 1;

    small_softmax_f32(data, scale, N);

    float sum = (N - 1) * std::exp(value * scale) + std::exp((value + 1) * scale);
    float v1 = std::exp((value + 1) * scale) / sum;
    float v2 = std::exp(value * scale) / sum;

    if (std::abs(data[0] - v1) > 0.001) return false;
    for (int i = 1; i < N; ++i) {
        if (std::abs(data[i] - v2) > 0.001) return false;
    }

    return true;
}

static bool test_softmax_bf16(int N, float scale) {
    float value = 1.0f;
    XDNN_BF16 data[N];
    for (int i = 0; i < N; ++i) {
        data[i] = (XDNN_BF16)value;
    }
    data[0] = (XDNN_BF16)(value + 1);

    small_softmax_bf16(data, scale, N);

    float sum = (N - 1) * std::exp(value * scale) + std::exp((value + 1) * scale);
    float v1 = std::exp((value + 1) * scale) / sum;
    float v2 = std::exp(value * scale) / sum;

    if (std::abs((float)data[0] - v1) > 0.001) return false;
    for (int i = 1; i < N; ++i) {
        if (std::abs((float)data[i] - v2) > 0.001) return false;
    }

    return true;
}

static void test(int N, float scale) {
    bool ret = test_softmax_f32(N, scale);
    if (ret) {
        printf("Passed: softmax_f32, N=%d, scale=%f\n", N, scale);
    } else {
        printf("Failed: softmax_f32, N=%d, scale=%f\n", N, scale);
    }

    ret = test_softmax_bf16(N, scale);
    if (ret) {
        printf("Passed: softmax_bf16, N=%d, scale=%f\n", N, scale);
    } else {
        printf("Failed: softmax_bf16, N=%d, scale=%f\n", N, scale);
    }
}

int main(int argc, char *argv[]) {
    test(32, 0.125f);
    test(32, 1.0f);
    test(128, 1.0f / sqrtf(128));

    return 0;
}