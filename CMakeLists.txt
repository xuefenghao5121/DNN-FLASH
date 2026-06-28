cmake_minimum_required(VERSION 3.16)
project(FlashOne LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLASHONE_ENABLE_NATIVE "Enable -march=native for local benchmarking" OFF)
option(FLASHONE_ENABLE_ONEDNN "Build optional oneDNN smoke/integration targets" ON)
option(FLASHONE_BUILD_TENSORFLOW_OP "Build TensorFlow custom op shared library" ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

add_library(flashone_core
  src/flashone/attention.cpp
  src/flashone/backend.cpp
  src/flashone/score_mod_plan.cpp
  src/flashone/runtime_plan.cpp
  src/flashone/tile_kernel.cpp
  src/flashone/attention_workspace.cpp
  src/flashone/batched_attention.cpp
)
target_include_directories(flashone_core PUBLIC include)
set_target_properties(flashone_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
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

add_executable(flashone_runtime_plan_tests tests/cpp/test_runtime_plan.cpp)
target_link_libraries(flashone_runtime_plan_tests PRIVATE flashone_core)

add_executable(flashone_q_tile_attention_tests tests/cpp/test_q_tile_attention.cpp)
target_link_libraries(flashone_q_tile_attention_tests PRIVATE flashone_core)

add_executable(flashone_qk_pv_tile_attention_tests tests/cpp/test_qk_pv_tile_attention.cpp)
target_link_libraries(flashone_qk_pv_tile_attention_tests PRIVATE flashone_core)

add_executable(flashone_workspace_attention_tests tests/cpp/test_workspace_attention.cpp)
target_link_libraries(flashone_workspace_attention_tests PRIVATE flashone_core)

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

    add_executable(flashone_onednn_postops_capability tests/cpp/test_onednn_postops_capability.cpp)
    target_link_libraries(flashone_onednn_postops_capability PRIVATE dnnl::dnnl)
    set_target_properties(flashone_onednn_postops_capability PROPERTIES
      BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/onednn-local/usr/lib/x86_64-linux-gnu")
  else()
    message(STATUS "oneDNN not found; skipping flashone_dnnl_smoke")
  endif()
endif()


if(FLASHONE_BUILD_TENSORFLOW_OP)
  execute_process(
    COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  execute_process(
    COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(TF_COMPILE_FLAGS_RAW AND TF_LINK_FLAGS_RAW)
    separate_arguments(TF_COMPILE_FLAGS UNIX_COMMAND "${TF_COMPILE_FLAGS_RAW}")
    separate_arguments(TF_LINK_FLAGS UNIX_COMMAND "${TF_LINK_FLAGS_RAW}")
    add_library(flashone_tf_attention SHARED tensorflow_ops/flashone_attention_op.cc)
    target_link_libraries(flashone_tf_attention PRIVATE flashone_core)
    target_compile_options(flashone_tf_attention PRIVATE ${TF_COMPILE_FLAGS})
    target_link_options(flashone_tf_attention PRIVATE ${TF_LINK_FLAGS})
    set_target_properties(flashone_tf_attention PROPERTIES
      PREFIX ""
      OUTPUT_NAME "flashone_tf_attention"
      BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/onednn-local/usr/lib/x86_64-linux-gnu")
  else()
    message(STATUS "TensorFlow compile/link flags unavailable; skipping flashone_tf_attention")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME flashone_cpp_tests COMMAND flashone_cpp_tests)
  add_test(NAME flashone_backend_tests COMMAND flashone_backend_tests)
  add_test(NAME flashone_tile_kernel_tests COMMAND flashone_tile_kernel_tests)
  add_test(NAME flashone_runtime_plan_tests COMMAND flashone_runtime_plan_tests)
  add_test(NAME flashone_q_tile_attention_tests COMMAND flashone_q_tile_attention_tests)
  add_test(NAME flashone_qk_pv_tile_attention_tests COMMAND flashone_qk_pv_tile_attention_tests)
  add_test(NAME flashone_workspace_attention_tests COMMAND flashone_workspace_attention_tests)
  if(TARGET flashone_dnnl_smoke)
    add_test(NAME flashone_dnnl_smoke COMMAND flashone_dnnl_smoke)
  endif()
  if(TARGET flashone_onednn_postops_capability)
    add_test(NAME flashone_onednn_postops_capability COMMAND flashone_onednn_postops_capability)
    set_tests_properties(flashone_onednn_postops_capability PROPERTIES
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endif()
endif()
