CXX = g++
GCCVERSIONGTEQ7 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 12)

CFLAGS = -O2 -std=c++17 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vbmi2 -mavx512fp16 -mavx512bf16 -fopenmp -Wno-ignored-attributes

all: check benchmark

check:
	@echo "CXX: " $(CXX)
	@echo "CXX VERSION: " $(shell expr `g++ -dumpversion`)
ifeq "$(GCCVERSIONGTEQ7)" "0"
	@echo "ERROR: GCC version MUST >= 12 to support 'constexpr if', according to https://en.cppreference.com/w/cpp/compiler_support#cpp17"
	exit
endif

benchmark: benchmark_sgemm benchmark_sgemm_f32f16f32 benchmark_sgemm_f32i8f32 benchmark_hgemm_f32f16f32 benchmark_hgemm_f32f16f16 benchmark_hgemm_f16f16f32 benchmark_hgemm_f32i8f32 benchmark_bgemm_f32bf16f32

benchmark_sgemm: benchmark_sgemm.cpp
	$(CXX) $(CFLAGS) benchmark_sgemm.cpp -o benchmark_sgemm -I../include -L../lib lxdnn

benchmark_sgemm_f32f16f32: benchmark_sgemm_f32f16f32.cpp
	$(CXX) $(CFLAGS) benchmark_sgemm_f32f16f32.cpp -o benchmark_sgemm_f32f16f32 -I../include -L../lib lxdnn

benchmark_sgemm_f32i8f32: benchmark_sgemm_f32i8f32.cpp
	$(CXX) $(CFLAGS) benchmark_sgemm_f32i8f32.cpp -o benchmark_sgemm_f32i8f32 -I../include -L../lib lxdnn

benchmark_hgemm_f32f16f32: benchmark_hgemm_f32f16f32.cpp
	$(CXX) $(CFLAGS) benchmark_hgemm_f32f16f32.cpp -o benchmark_hgemm_f32f16f32 -I../include -L../lib lxdnn

benchmark_hgemm_f32f16f16: benchmark_hgemm_f32f16f16.cpp
	$(CXX) $(CFLAGS) benchmark_hgemm_f32f16f16.cpp -o benchmark_hgemm_f32f16f16 -I../include -L../lib lxdnn

benchmark_hgemm_f16f16f32: benchmark_hgemm_f16f16f32.cpp
	$(CXX) $(CFLAGS) benchmark_hgemm_f16f16f32.cpp -o benchmark_hgemm_f16f16f32 -I../include -L../lib lxdnn

benchmark_hgemm_f32i8f32: benchmark_hgemm_f32i8f32.cpp
	$(CXX) $(CFLAGS) benchmark_hgemm_f32i8f32.cpp -o benchmark_hgemm_f32i8f32 -I../include -L../lib lxdnn

benchmark_bgemm_f32bf16f32: benchmark_bgemm_f32bf16f32.cpp
	$(CXX) $(CFLAGS) benchmark_bgemm_f32bf16f32.cpp -o benchmark_bgemm_f32bf16f32 -I../include -L../lib lxdnn

clean:
	rm -f benchmark_sgemm benchmark_sgemm_f32f16f32 benchmark_sgemm_f32i8f32 benchmark_hgemm_f32f16f32 benchmark_hgemm_f32f16f16 benchmark_hgemm_f16f16f32 benchmark_hgemm_f32i8f32 benchmark_bgemm_f32bf16f32
