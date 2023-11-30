#!/bin/bash
set -x

LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_sgemm
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_sgemm_f32f16f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_sgemm_f32i8f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_hgemm_f32f16f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_hgemm_f32f16f16
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_hgemm_f16f16f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_hgemm_f32i8f32
LD_LIBRARY_PATH=../lib/:$LD_LIBRARY_PATH numactl -C 48-95 -m 3 ./test_bgemm_f32bf16f32

