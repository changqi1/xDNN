#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <memory>

#include "data_types/data_types.h"
#include "intrinsic_ext.h"
#include "../utils/utils.h"

#define ALLOC(DATATYPE, VALUE, SIZE)  std::unique_ptr<DATATYPE, decltype(&free)> VALUE(static_cast<DATATYPE*>(aligned_alloc(64, SIZE * sizeof(DATATYPE))), &free)

void test_xdnn_intrinsic_ext_bf16_load_store(int size) {
    ALLOC(XDNN_BF16, src, size);
    ALLOC(XDNN_BF16, dst, size);

    test_utils::init(src.get(), size, -0.25f, 0.25f);
    test_utils::init(dst.get(), size, -0.11f);

    int block_num = size / AVX2_BF16_NUM;
    int remain = size - block_num * AVX2_BF16_NUM;

    for (int i = 0; i < block_num; ++i) {
        __m512 val = _mm512_loadu_pbh(src.get() + i * AVX2_BF16_NUM);
        val = val - val;
        _mm512_storeu_pbh(dst.get() + i * AVX2_BF16_NUM, val);
    }

    if (remain > 0) {
        __mmask16 mask = (remain >= AVX2_BF16_NUM ? 0xffff : (1 << remain) - 1);
        __m512 val = _mm512_maskz_loadu_pbh(mask, src.get() + block_num * AVX2_BF16_NUM);
        val = val - val;
        _mm512_mask_storeu_pbh(dst.get() + block_num * AVX2_BF16_NUM, mask, val);
    }

    for (int i = 0; i < size; ++i) {
        if (dst.get()[i] != 0.0f) {
            printf("\tFailed: bf16_load_store (size=%d, index=%d)\n", size, i);
            return;
        }
    }
    printf("\tPassed: size=%d\n", size);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    printf("Test xdnn_intrinsic_ext_bf16_load_store:\n");
    test_xdnn_intrinsic_ext_bf16_load_store(68);

    return 0;
}
