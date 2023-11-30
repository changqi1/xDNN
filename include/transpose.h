#pragma once

#include <immintrin.h>

#include "data_types/data_types.h"

void xdnn_transpose(const float *src, int src_rows, int src_cols, int src_stride, float *dst, int dst_stride);
void xdnn_transpose(const XDNN_BF16 *src, int src_rows, int src_cols, int src_stride, XDNN_BF16 *dst, int dst_stride);
void xdnn_transpose(const int *src, int src_rows, int src_cols, int src_stride, int *dst, int dst_stride);

void xdnn_transpose_16x16_v1(const int32_t *src, int src_stride, int32_t *dst, int dst_stride);
void xdnn_transpose_16x16_v2(const int32_t *src, int src_stride, int32_t *dst, int dst_stride);
void xdnn_transpose_16xN_v1(const int32_t *src, int cols, int src_stride, int32_t *dst, int dst_stride);

void xdnn_transpose16x32_packBA16a16b2a_v1(const XDNN_BF16 *src, int src_stride, XDNN_BF16 *dst, int dst_stride);
void xdnn_transpose16x32_packBA16a16b2a_v2(const XDNN_BF16 *src, int src_stride, XDNN_BF16 *dst, int dst_stride);
void xdnn_transpose16xN_packBA16a16b2a_v1(const XDNN_BF16 *src, int src_cols, int src_stride, XDNN_BF16 *dst, int dst_rows, int dst_stride);