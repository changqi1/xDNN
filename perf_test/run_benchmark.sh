#!/bin/bash
set -x

LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./benchmark_sgemm
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./benchmark_sgemm_f32f16f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./benchmark_sgemm_f32s8f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./benchmark_hgemm_f32f16f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./benchmark_hgemm_f32s8f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./benchmark_bgemm_f32bf16f32

