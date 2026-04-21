cmake_minimum_required(VERSION 3.18)
project(cudasearch CXX)

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ─────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Options ────────────────────────────────────────────────────────────────────
option(USE_CUDA   "Build CUDA kernels"        OFF)
option(USE_MPI    "Build MPI distributed code" OFF)
option(USE_OPENMP "Enable OpenMP"              ON)

# ── OpenMP ─────────────────────────────────────────────────────────────────────
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# ── CUDA ───────────────────────────────────────────────────────────────────────
if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  # Set to your target GPU: 80=A100, 86=RTX3090, 89=RTX4090, 90=H100
  # Check with: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ── MPI ────────────────────────────────────────────────────────────────────────
if(USE_MPI)
  find_package(MPI REQUIRED)
endif()

# ── Compiler flags ─────────────────────────────────────────────────────────────
add_compile_options(-O3 -march=native -Wall -Wextra)

# ── Library: io ────────────────────────────────────────────────────────────────
add_library(cudasearch_io STATIC
  src/io/fvecs_loader.cpp
)
target_include_directories(cudasearch_io PUBLIC src)

# ── Library: core (CPU search + recall) ────────────────────────────────────────
add_library(cudasearch_core STATIC
  src/core/cpu_search.cpp
)
target_include_directories(cudasearch_core PUBLIC src)
if(USE_OPENMP)
  target_link_libraries(cudasearch_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Library: cuda ──────────────────────────────────────────────────────────────
if(USE_CUDA)
  add_library(cudasearch_cuda STATIC
    src/cuda/gpu_search.cu
  )
  target_include_directories(cudasearch_cuda PUBLIC src)
  target_link_libraries(cudasearch_cuda PUBLIC cudasearch_core)
  target_compile_definitions(cudasearch_cuda PUBLIC HAVE_CUDA)
endif()

# ── Executable: bench ──────────────────────────────────────────────────────────
add_executable(bench bench/main_bench.cpp)
target_link_libraries(bench PRIVATE cudasearch_core cudasearch_io)
if(USE_OPENMP)
  target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_CUDA)
  target_link_libraries(bench PRIVATE cudasearch_cuda)
endif()

# ── Executable: test_recall ────────────────────────────────────────────────────
add_executable(test_recall tests/test_recall.cpp)
target_link_libraries(test_recall PRIVATE cudasearch_core cudasearch_io)
if(USE_OPENMP)
  target_link_libraries(test_recall PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_CUDA)
  target_link_libraries(test_recall PRIVATE cudasearch_cuda)
endif()

# ── Summary ────────────────────────────────────────────────────────────────────
message(STATUS "Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "USE_CUDA    : ${USE_CUDA}")
message(STATUS "USE_MPI     : ${USE_MPI}")
message(STATUS "USE_OPENMP  : ${USE_OPENMP}")
