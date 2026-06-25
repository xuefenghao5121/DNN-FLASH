cmake_minimum_required(VERSION 3.16)
project(FlashOne LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLASHONE_ENABLE_NATIVE "Enable -march=native for local benchmarking" OFF)

add_library(flashone_core
  src/flashone/attention.cpp
  src/flashone/backend.cpp
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

add_executable(flashone_bench benchmarks/bench_attention.cpp)
target_link_libraries(flashone_bench PRIVATE flashone_core)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME flashone_cpp_tests COMMAND flashone_cpp_tests)
  add_test(NAME flashone_backend_tests COMMAND flashone_backend_tests)
endif()
