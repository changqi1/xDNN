#pragma once

#include "data_types/data_types.h"

extern "C" {
void small_softmax_f32(float *data, const float scale, int size);
void small_softmax_bf16(XDNN_BF16 *data, const float scale, int size);
}