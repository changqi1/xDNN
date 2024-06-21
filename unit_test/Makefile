CXX = g++
GCCVERSIONGTEQ7 := $(shell expr `g++ -dumpversion | cut -f1 -d.` \>= 12)

CFLAGS = -O2 -std=c++17 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vbmi2 -mavx512fp16 -mavx512bf16 -fopenmp -Wno-ignored-attributes

all: check test

check:
	@echo "CXX: " $(CXX)
	@echo "CXX VERSION: " $(shell expr `g++ -dumpversion`)
ifeq "$(GCCVERSIONGTEQ7)" "0"
	@echo "ERROR: GCC version MUST >= 12 to support 'constexpr if', according to https://en.cppreference.com/w/cpp/compiler_support#cpp17"
	exit
endif

test: test_sgemm test_sgemm_f32f16f32 test_sgemm_f32s8f32 test_hgemm_f32f16f32 test_hgemm_f32f16f16 test_hgemm_f16f16f32 test_hgemm_f32s8f32 test_bgemm_f32bf16f32

test_sgemm: test_sgemm.cpp
	$(CXX) $(CFLAGS) test_sgemm.cpp -o test_sgemm -I../include -L../lib -lxdnn

test_sgemm_f32f16f32: test_sgemm_f32f16f32.cpp
	$(CXX) $(CFLAGS) test_sgemm_f32f16f32.cpp -o test_sgemm_f32f16f32 -I../include -L../lib -lxdnn

test_sgemm_f32s8f32: test_sgemm_f32s8f32.cpp
	$(CXX) $(CFLAGS) test_sgemm_f32s8f32.cpp -o test_sgemm_f32s8f32 -I../include -L../lib -lxdnn

test_hgemm_f32f16f32: test_hgemm_f32f16f32.cpp
	$(CXX) $(CFLAGS) test_hgemm_f32f16f32.cpp -o test_hgemm_f32f16f32 -I../include -L../lib -lxdnn

test_hgemm_f32f16f16: test_hgemm_f32f16f16.cpp
	$(CXX) $(CFLAGS) test_hgemm_f32f16f16.cpp -o test_hgemm_f32f16f16 -I../include -L../lib -lxdnn

test_hgemm_f16f16f32: test_hgemm_f16f16f32.cpp
	$(CXX) $(CFLAGS) test_hgemm_f16f16f32.cpp -o test_hgemm_f16f16f32 -I../include -L../lib -lxdnn

test_hgemm_f32s8f32: test_hgemm_f32s8f32.cpp
	$(CXX) $(CFLAGS) test_hgemm_f32s8f32.cpp -o test_hgemm_f32s8f32 -I../include -L../lib -lxdnn

test_bgemm_f32bf16f32: test_bgemm_f32bf16f32.cpp
	$(CXX) $(CFLAGS) test_bgemm_f32bf16f32.cpp -o test_bgemm_f32bf16f32 -I../include -L../lib -lxdnn

clean:
	rm -f test_sgemm test_sgemm_f32f16f32 test_sgemm_f32s8f32 test_hgemm_f32f16f32 test_hgemm_f32f16f16 test_hgemm_f16f16f32 test_hgemm_f32s8f32 test_bgemm_f32bf16f32
