cmake_minimum_required(VERSION 3.16)
project(FlashOne LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ONEDNN_FLASH_ENABLE_NATIVE "Enable -march=native for local benchmarking" OFF)
option(ONEDNN_FLASH_ENABLE_ONEDNN "Build optional oneDNN smoke/integration targets" ON)
option(ONEDNN_FLASH_BUILD_TENSORFLOW_OP "Build TensorFlow custom op shared library" ON)
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE ONEDNN_FLASH_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT ONEDNN_FLASH_GIT_COMMIT)
  set(ONEDNN_FLASH_GIT_COMMIT "unknown")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

add_library(onednn_flash_core
  src/onednn_flash/attention.cpp
  src/onednn_flash/backend.cpp
  src/onednn_flash/score_mod_plan.cpp
  src/onednn_flash/runtime_plan.cpp
  src/onednn_flash/qk_score_tile.cpp
  src/onednn_flash/tile_kernel.cpp
  src/onednn_flash/attention_workspace.cpp
  src/onednn_flash/batched_attention.cpp
)
target_include_directories(onednn_flash_core PUBLIC include PRIVATE src)
set_target_properties(onednn_flash_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(onednn_flash_core PRIVATE -Wall -Wextra -Wpedantic)
if(ONEDNN_FLASH_ENABLE_NATIVE)
  target_compile_options(onednn_flash_core PRIVATE -march=native)
endif()

add_executable(onednn_flash_bench benchmarks/bench_attention.cpp)
target_link_libraries(onednn_flash_bench PRIVATE onednn_flash_core)

add_executable(onednn_flash_qk_postops_bench benchmarks/bench_qk_postops.cpp)
target_link_libraries(onednn_flash_qk_postops_bench PRIVATE onednn_flash_core)
target_include_directories(onednn_flash_qk_postops_bench PRIVATE src)
target_compile_definitions(onednn_flash_qk_postops_bench PRIVATE ONEDNN_FLASH_GIT_COMMIT="${ONEDNN_FLASH_GIT_COMMIT}")

if(ONEDNN_FLASH_ENABLE_ONEDNN)
  find_package(LocalDnnl QUIET)
  if(DNNL_FOUND)
    target_sources(onednn_flash_core PRIVATE src/onednn_flash/onednn_tile_kernel.cpp)
    target_compile_definitions(onednn_flash_core PUBLIC ONEDNN_FLASH_HAS_ONEDNN=1)
    target_link_libraries(onednn_flash_core PUBLIC dnnl::dnnl)

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${DNNL_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${DNNL_LIBRARY}")
    set(CMAKE_REQUIRED_DEFINITIONS "-DDNNL_EXPERIMENTAL_UKERNEL")
    check_cxx_source_compiles("
      #include <oneapi/dnnl/dnnl.hpp>
      #include <oneapi/dnnl/dnnl_ukernel.hpp>
      int main() {
        auto pack = dnnl::ukernel::brgemm::get_B_pack_type(
          dnnl::memory::data_type::f32, dnnl::memory::data_type::f32);
        return pack == dnnl::ukernel::pack_type::undef;
      }
    " ONEDNN_FLASH_ONEDNN_BRGEMM_COMPILES)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    unset(CMAKE_REQUIRED_DEFINITIONS)

    if(ONEDNN_FLASH_ONEDNN_BRGEMM_COMPILES)
      target_sources(onednn_flash_core PRIVATE src/onednn_flash/onednn_brgemm_tile_kernel.cpp)
      target_compile_definitions(onednn_flash_core PUBLIC ONEDNN_FLASH_HAS_ONEDNN_BRGEMM=1 DNNL_EXPERIMENTAL_UKERNEL)
      message(STATUS "oneDNN BRGEMM ukernel API found; enabling FlashOne BRGEMM tile kernel")
    else()
      message(STATUS "oneDNN BRGEMM ukernel API not found; BRGEMM tile kernel disabled")
      if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/onednn-debian-3.12/usr/include/oneapi/dnnl/dnnl_ukernel.hpp")
        message(STATUS "Hint: if BRGEMM was unexpectedly disabled in an existing build directory, clear cached DNNL_INCLUDE_DIR/DNNL_LIBRARY or configure a fresh build directory")
      endif()
    endif()


    # Stage 1 minimal validation test (standalone, no dependencies on onednn_flash_core)
    add_executable(onednn_flash_stage1_minimal tests/cpp/test_stage1_minimal.cpp)
    target_link_libraries(onednn_flash_stage1_minimal PRIVATE dnnl::dnnl)
    set_target_properties(onednn_flash_stage1_minimal PROPERTIES
      BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/onednn-local/usr/lib/x86_64-linux-gnu;${CMAKE_SOURCE_DIR}/third_party/onednn-debian-3.12/usr/lib/x86_64-linux-gnu")
    message(STATUS "Stage 1 minimal validation test enabled")
  else()
    message(STATUS "oneDNN not found; skipping onednn_flash_dnnl_smoke")
  endif()
endif()


if(ONEDNN_FLASH_BUILD_TENSORFLOW_OP)
  execute_process(
    COMMAND python3 -c "import sys; import tensorflow as tf; sys.stderr = open('/dev/null', 'w'); print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND python3 -c "import sys; import tensorflow as tf; sys.stderr = open('/dev/null', 'w'); print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "TensorFlow compile flags: ${TF_COMPILE_FLAGS_RAW}")
  message(STATUS "TensorFlow link flags: ${TF_LINK_FLAGS_RAW}")
  if(TF_COMPILE_FLAGS_RAW AND TF_LINK_FLAGS_RAW)
    separate_arguments(TF_COMPILE_FLAGS UNIX_COMMAND "${TF_COMPILE_FLAGS_RAW}")
    separate_arguments(TF_LINK_FLAGS UNIX_COMMAND "${TF_LINK_FLAGS_RAW}")
    add_library(onednn_flash_tf_attention SHARED tensorflow_ops/onednn_flash_attention_op.cc)
    target_link_libraries(onednn_flash_tf_attention PRIVATE onednn_flash_core)
    target_compile_options(onednn_flash_tf_attention PRIVATE ${TF_COMPILE_FLAGS})
    target_link_options(onednn_flash_tf_attention PRIVATE ${TF_LINK_FLAGS})
    set_target_properties(onednn_flash_tf_attention PROPERTIES
      PREFIX ""
      OUTPUT_NAME "onednn_flash_tf_attention"
      BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/onednn-local/usr/lib/x86_64-linux-gnu;${CMAKE_SOURCE_DIR}/third_party/onednn-debian-3.12/usr/lib/x86_64-linux-gnu")
  else()
    message(STATUS "TensorFlow compile/link flags unavailable; skipping onednn_flash_tf_attention")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  if(TARGET onednn_flash_stage1_minimal)
    add_test(NAME onednn_flash_stage1_minimal COMMAND onednn_flash_stage1_minimal)
  endif()
endif()
