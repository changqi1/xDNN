cmake_minimum_required(VERSION 3.15.1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(test_intrinsic_ext test_intrinsic_ext.cpp)
target_link_libraries(test_intrinsic_ext PRIVATE ${XDNN_LIB})

add_executable(test_transpose test_transpose.cpp)
target_link_libraries(test_transpose PRIVATE ${XDNN_LIB})

add_executable(test_sgemm test_sgemm.cpp)
target_link_libraries(test_sgemm PRIVATE ${XDNN_LIB})

add_executable(test_sgemm_f32f16f32 test_sgemm_f32f16f32.cpp)
target_link_libraries(test_sgemm_f32f16f32 PRIVATE ${XDNN_LIB})

add_executable(test_sgemm_f32i8f32 test_sgemm_f32i8f32.cpp)
target_link_libraries(test_sgemm_f32i8f32 PRIVATE ${XDNN_LIB})

add_executable(test_hgemm_f16f16f32 test_hgemm_f16f16f32.cpp)
target_link_libraries(test_hgemm_f16f16f32 PRIVATE ${XDNN_LIB})

add_executable(test_hgemm_f32f16f16 test_hgemm_f32f16f16.cpp)
target_link_libraries(test_hgemm_f32f16f16 PRIVATE ${XDNN_LIB})

add_executable(test_hgemm_f32f16f32 test_hgemm_f32f16f32.cpp)
target_link_libraries(test_hgemm_f32f16f32 PRIVATE ${XDNN_LIB})

add_executable(test_hgemm_f32i8f32 test_hgemm_f32i8f32.cpp)
target_link_libraries(test_hgemm_f32i8f32 PRIVATE ${XDNN_LIB})

add_executable(test_bgemm_f32bf16f32 test_bgemm_f32bf16f32.cpp)
target_link_libraries(test_bgemm_f32bf16f32 PRIVATE ${XDNN_LIB})

add_executable(test_amx_sgemm_bf16bf16bf16 test_amx_sgemm_bf16bf16bf16.cpp)
target_link_libraries(test_amx_sgemm_bf16bf16bf16 PRIVATE ${XDNN_LIB})