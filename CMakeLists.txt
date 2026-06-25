cmake_minimum_required(VERSION 3.16)
project(FlashOne LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLASHONE_ENABLE_NATIVE "Enable -march=native for local benchmarking" OFF)
option(FLASHONE_ENABLE_ONEDNN "Build optional oneDNN smoke/integration targets" ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

add_library(flashone_core
  src/flashone/attention.cpp
  src/flashone/backend.cpp
  src/flashone/tile_kernel.cpp
)
target_include_directories(flashone_core PUBLIC include)
target_compile_options(flashone_core PRIVATE -Wall -Wextra -Wpedantic)
if(FLASHONE_ENABLE_NATIVE)
  target_compile_options(flashone_core PRIVATE -march=native)
endif()

add_executable(flashone_cpp_tests tests/cpp/test_attention.cpp)
target_link_libraries(flashone_cpp_tests PRIVATE flashone_core)

add_executable(flashone_backend_tests tests/cpp/test_backend.cpp)
target_link_libraries(flashone_backend_tests PRIVATE flashone_core)

add_executable(flashone_tile_kernel_tests tests/cpp/test_tile_kernel.cpp)
target_link_libraries(flashone_tile_kernel_tests PRIVATE flashone_core)

add_executable(flashone_q_tile_attention_tests tests/cpp/test_q_tile_attention.cpp)
target_link_libraries(flashone_q_tile_attention_tests PRIVATE flashone_core)

add_executable(flashone_qk_pv_tile_attention_tests tests/cpp/test_qk_pv_tile_attention.cpp)
target_link_libraries(flashone_qk_pv_tile_attention_tests PRIVATE flashone_core)

add_executable(flashone_bench benchmarks/bench_attention.cpp)
target_link_libraries(flashone_bench PRIVATE flashone_core)

if(FLASHONE_ENABLE_ONEDNN)
  find_package(LocalDnnl QUIET)
  if(DNNL_FOUND)
    target_sources(flashone_core PRIVATE src/flashone/onednn_tile_kernel.cpp)
    target_compile_definitions(flashone_core PUBLIC FLASHONE_HAS_ONEDNN=1)
    target_link_libraries(flashone_core PUBLIC dnnl::dnnl)

    add_executable(flashone_dnnl_smoke tests/cpp/test_dnnl_smoke.cpp)
    target_link_libraries(flashone_dnnl_smoke PRIVATE dnnl::dnnl)
    set_target_properties(flashone_dnnl_smoke PROPERTIES
      BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/onednn-local/usr/lib/x86_64-linux-gnu")
  else()
    message(STATUS "oneDNN not found; skipping flashone_dnnl_smoke")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME flashone_cpp_tests COMMAND flashone_cpp_tests)
  add_test(NAME flashone_backend_tests COMMAND flashone_backend_tests)
  add_test(NAME flashone_tile_kernel_tests COMMAND flashone_tile_kernel_tests)
  add_test(NAME flashone_q_tile_attention_tests COMMAND flashone_q_tile_attention_tests)
  add_test(NAME flashone_qk_pv_tile_attention_tests COMMAND flashone_qk_pv_tile_attention_tests)
  if(TARGET flashone_dnnl_smoke)
    add_test(NAME flashone_dnnl_smoke COMMAND flashone_dnnl_smoke)
  endif()
endif()
