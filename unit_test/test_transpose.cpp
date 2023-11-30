#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <memory>

#include "transpose.h"
#include "../utils/utils.h"

#define ACCURACY 0.000001f

void test_xdnn_transpose_int32(int rows, int cols) {
    ALLOC(int32_t, B, rows * cols);
    ALLOC(int32_t, transposedB1, cols * rows);
    ALLOC(int32_t, transposedB2, rows * cols);

    test_utils::init(B.get(), rows * cols, -1000, 1000);

    // test_utils::print("B", B.get(), rows, cols, cols);
    xdnn_transpose_16x16_v1(B.get(), cols, transposedB1.get(), rows);
    // test_utils::print("transposedB1", transposedB1.get(), cols/2, rows*2, rows*2);
    xdnn_transpose_16x16_v1(transposedB1.get(), cols, transposedB2.get(), rows);
    // test_utils::print("transposedB2", transposedB2.get(), rows, cols, cols);

    test_utils::validate(rows * cols, B.get(), transposedB2.get(), ACCURACY);
}

void test_xdnn_transpose_pack_bf16(int rows, int cols) {
    ALLOC(XDNN_BF16, B, rows * cols);
    ALLOC(XDNN_BF16, transposedB1, ((cols+31)/32)*32 * rows);
    ALLOC(XDNN_BF16, transposedB2, rows * cols);

    test_utils::init(B.get(), rows * cols, -0.25f, 0.25f);

    // test_utils::print("B", B.get(), rows, cols, cols);
    xdnn_transpose16x32_packBA16a16b2a_v1(B.get(), cols, transposedB1.get(), rows);
    // test_utils::print("transposedB1", transposedB1.get(), rows, cols, cols);
    xdnn_transpose16x32_packBA16a16b2a_v2(transposedB1.get(), cols, transposedB2.get(), rows);
    // test_utils::print("transposedB2", transposedB2.get(), rows, cols, cols);

    test_utils::validate(rows * cols, B.get(), transposedB2.get(), ACCURACY);

    // test_utils::print("B", B.get(), rows, cols, cols);
    xdnn_transpose16xN_packBA16a16b2a_v1(B.get(), cols, cols, transposedB1.get(), 16, 32);
    // test_utils::print("transposedB1", transposedB1.get(), rows, ((cols+31)/32)*32, ((cols+31)/32)*32);
    xdnn_transpose16xN_packBA16a16b2a_v1(transposedB1.get(), cols, cols, transposedB2.get(), 16, 32);
    // test_utils::print("transposedB2", transposedB2.get(), rows, cols, cols);

    test_utils::validate(rows * cols, B.get(), transposedB2.get(), ACCURACY);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc == 3) {
        int rows = std::stoi(argv[1]);
        int cols = std::stoi(argv[2]);

        test_xdnn_transpose_pack_bf16(rows, cols);

        return 0;
    }

    printf("Test xdnn_transpose:\n");
    test_xdnn_transpose_int32(16, 16);

    printf("Test xdnn_transpose_pack_bf16:\n");
    test_xdnn_transpose_pack_bf16(16, 32);

    return 0;
}
