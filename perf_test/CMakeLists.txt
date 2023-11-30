cmake_minimum_required(VERSION 3.15.1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(benchmark_sgemm benchmark_sgemm.cpp)
target_link_libraries(benchmark_sgemm PRIVATE ${XDNN_LIB})

add_executable(benchmark_sgemm_f32f16f32 benchmark_sgemm_f32f16f32.cpp)
target_link_libraries(benchmark_sgemm_f32f16f32 PRIVATE ${XDNN_LIB})

add_executable(benchmark_sgemm_f32i8f32 benchmark_sgemm_f32i8f32.cpp)
target_link_libraries(benchmark_sgemm_f32i8f32 PRIVATE ${XDNN_LIB})

add_executable(benchmark_hgemm_f16f16f32 benchmark_hgemm_f16f16f32.cpp)
target_link_libraries(benchmark_hgemm_f16f16f32 PRIVATE ${XDNN_LIB})

add_executable(benchmark_hgemm_f32f16f16 benchmark_hgemm_f32f16f16.cpp)
target_link_libraries(benchmark_hgemm_f32f16f16 PRIVATE ${XDNN_LIB})

add_executable(benchmark_hgemm_f32f16f32 benchmark_hgemm_f32f16f32.cpp)
target_link_libraries(benchmark_hgemm_f32f16f32 PRIVATE ${XDNN_LIB})

add_executable(benchmark_hgemm_f32i8f32 benchmark_hgemm_f32i8f32.cpp)
target_link_libraries(benchmark_hgemm_f32i8f32 PRIVATE ${XDNN_LIB})

add_executable(benchmark_bgemm_f32bf16f32 benchmark_bgemm_f32bf16f32.cpp)
target_link_libraries(benchmark_bgemm_f32bf16f32 PRIVATE ${XDNN_LIB})

add_executable(benchmark_amx_sgemm_bf16bf16bf16 benchmark_amx_sgemm_bf16bf16bf16.cpp)
target_link_libraries(benchmark_amx_sgemm_bf16bf16bf16 PRIVATE ${XDNN_LIB})