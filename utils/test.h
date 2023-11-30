#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include <cassert>
#include <type_traits>
#include <tuple>

#include <immintrin.h>

// #include "../include/common/common.h"

class test_utils {
public:
  // A: M x K; B: K x N; C: M x N;
  template<typename TA, typename TB, typename TC>
  static void gemm_ref(bool transA, bool transB, int M, int N, int K,
        float alpha, const TA *A, int lda, const TB *B, int ldb,
        float beta, TC *C, int ldc) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        // C[i,j] = SUM(A[i,k] * B[k,j])
        TC sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          sum += (TC)(A[i * lda + k] * B[k * ldb + j]);
#if DEBUG
          printf("%d %d %d: %f %f %f\n", i, j, k, A[i * lda + k], float(B[k * ldb + j]), sum);
#endif
        }
        C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
      }
    }
  }

  template<typename TA, typename TB, typename TC, typename TSB>
  static void gemm_ref(bool transA, bool transB, int M, int N, int K,
        float alpha, const TA *A, int lda, const TB *B, int ldb, const TSB *scaleB, const TSB *zeroB,
        float beta, TC *C, int ldc) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        // C[i,j] = SUM(A[i,k] * B[k,j])
        TC sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          sum += (TSB)(A[i * lda + k] * (B[k * ldb + j]  * scaleB[j] + zeroB[j]));
        }
        C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
      }
    }
  }

  template<typename TC, typename Tbias>
  static void add_bias(int M, int N, TC *C, int ldc, const Tbias *bias) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += bias[j];
      }
    }
  }

  // Transpose B into transposedB
  // B: shape is K x N
  // transposedB: shape is N x K
  template<typename TB>
  static void transpose(int N, int K, const TB *B, int ldb, TB *transposedB) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < K; ++j) {
        // transposedB[i, j] = B[j, i];
        transposedB[i * K + j] = B[j * N + i];
      }
    }
  }

  template<typename TC>
  static bool is_same_matrix(int M, int N, const TC *refC, const TC *C, int ldc, float threshold) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs(refC[i * ldc + j] - C[i * ldc + j]) > threshold &&
            fabs((refC[i * ldc + j] - C[i * ldc + j]) / refC[i * ldc + j]) > threshold) {
          return false;
        }
      }
    }
    return true;
  }

  template<typename TCR, typename TC>
  static std::tuple<int, int, float, float, float> diff_index(int M, int N, const TCR *refC, const TC *C, int ldc, float threshold) {
    int first_diff_index = -1;
    int error_count = 0;
    float absolute_error_sum = 0.0;
    float relative_error_sum = 0.0;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float absolute_error = fabs((float)refC[i * ldc + j] - (float)C[i * ldc + j]);
        float relative_error = fabs(((float)refC[i * ldc + j] - (float)C[i * ldc + j]) / (float)refC[i * ldc + j]);
        if (absolute_error > threshold && relative_error > threshold) {
          if (first_diff_index == -1) {
            first_diff_index = i * ldc + j;
          }
          error_count++;
          absolute_error_sum += absolute_error;
          relative_error_sum += relative_error;
        }
      }
    }

    if (error_count == 0) {
      return std::make_tuple(-1, 0, 0.0, 0.0, 0.0);
    }
    return std::make_tuple(first_diff_index, error_count, (float)error_count/(M*N),
                           absolute_error_sum/error_count, relative_error_sum/error_count);
  }

  template<typename TCR, typename TC>
  static void validate(int M, int N, int K, int lda, int ldb, int ldc, const TCR *refC, const TC *C, float threshold) {
    std::tuple<int, int, float, float, float> ret = diff_index(M, N, refC, C, ldc, threshold);
    int idx = std::get<0>(ret);
    if (idx != -1) {
        printf("\tFailed: M=%5d, N=%5d, K=%5d, lda=%5d, ldb=%5d, ldc=%5d, ref[%d]=%.6f, our[%d]=%.6f, "
               "error_count=%d, \terror_rate=%.2f% with absolute_error=%.6f, relative_error=%.2f%\n",
               M, N, K, lda, ldb, ldc, idx, float(refC[idx]), idx, float(C[idx]),
               std::get<1>(ret), std::get<2>(ret)*100, std::get<3>(ret), std::get<4>(ret)*100);
    } else {
        printf("\tPassed: M=%5d, N=%5d, K=%5d, lda=%5d, ldb=%5d, ldc=%5d\n", M, N, K, lda, ldb, ldc);
    }
  }

  template<typename TCR, typename TC>
  static void validate(int size, const TCR *refC, const TC *C, float threshold) {
    std::tuple<int, int, float, float, float> ret = diff_index(1, size, refC, C, size, threshold);
    int idx = std::get<0>(ret);
    if (idx != -1) {
        printf("\tFailed: size=%5d, ref[%d]=%.6f, our[%d]=%.6f, "
               "error_count=%d, \terror_rate=%.2f% with absolute_error=%.6f, relative_error=%.2f%\n",
               size, idx, float(refC[idx]), idx, float(C[idx]),
               std::get<1>(ret), std::get<2>(ret)*100, std::get<3>(ret), std::get<4>(ret)*100);
    } else {
        printf("\tPassed: size=%5d\n", size);
    }
  }

  template<typename T>
  static void print(const char *name, T &v) {
    printf("%s: ", name);
    if constexpr (std::is_same<T, __m512h>::value) {
      XDNN_FP16 val[32] = { 0.0f };
      _mm512_storeu_ph(val, v);
      for (int i = 0; i < 32; ++i) {
        printf("%.6f, ", float(val[i]));
      }
      printf("\n");
    }
    else if constexpr (std::is_same<T, __m512>::value) {
      float val[16] = { 0.0f };
      _mm512_storeu_ps(val, v);
      for (int i = 0; i < 16; ++i) {
        printf("%.6f, ", val[i]);
      }
      printf("\n");
    }
  }

  template<typename T>
  static void print(const char *name, T *data, int rows, int cols, int stride) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%f, ", float(data[i*stride+j]));
      }
      printf("\n");
    }
    printf("\n");
  }

  template<typename T>
  static void print(const char *name, T *data, int size) {
    printf("%s:\n", name);
    for (int i = 0; i < size; ++i) {
      printf("%f, ", float(data[i]));
    }
    printf("\n");
  }

  template<typename T>
  static void init(T *buf, int size, float init_val) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = static_cast<T>(init_val);
    }
  }

  template<typename T>
  static void init(T *buf, int size, float min_value, float max_value) {
    // std::default_random_engine generator(static_cast<unsigned>(size));
    // std::uniform_real_distribution<float> distribution(static_cast<float>(min_value), static_cast<float>(max_value));
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, static_cast<float>(max_value/2));

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float data = distribution(generator);
        if (data > max_value)
          data = max_value;
        if (data < min_value)
          data = min_value;
        buf[i] = static_cast<T>(data);
    }
  }
};
/*
template<>
void test_utils::init<XDNN_FP16>(XDNN_FP16 *buf, int size, XDNN_FP16 init_val) {
  assert(size % 4 == 0);
  std::default_random_engine generator(static_cast<unsigned>(size));
  std::uniform_real_distribution<float> distribution(static_cast<float>(-0.25f), static_cast<float>(0.25f));

  #pragma omp parallel for
  for (int i = 0; i < size; i += 4) {
    if (init_val == (float)0.0f) {
      float val[4] = {static_cast<float>(distribution(generator)),
                      static_cast<float>(distribution(generator)),
                      static_cast<float>(distribution(generator)),
                      static_cast<float>(distribution(generator))};
      _mm_storeu_ph(&buf[i], _mm_cvtxps_ph(_mm_loadu_ps(val)));
    }
    else {
      buf[i] = static_cast<float>(init_val);
      buf[i + 1] = static_cast<float>(init_val);
      buf[i + 2] = static_cast<float>(init_val);
      buf[i + 3] = static_cast<float>(init_val);
    }
  }
}
*/