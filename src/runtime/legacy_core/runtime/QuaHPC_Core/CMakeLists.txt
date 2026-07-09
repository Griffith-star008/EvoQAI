cmake_minimum_required(VERSION 3.20)
project(qhpc LANGUAGES CXX CUDA)

# ----- Options -----
option(QHPC_BUILD_PYTHON "Build Python (pybind11) module" ON)
option(QHPC_HAS_NVSHMEM  "Build with NVSHMEM distributed backend" OFF)
set(QHPC_SM_ARCH "89" CACHE STRING "CUDA SM architecture (e.g. 80, 86, 89, 90)")

# ----- C++ + CUDA flags -----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${QHPC_SM_ARCH})

# -fPIC is meaningless and forbidden by MSVC on Windows. Only enable on UNIX.
if(UNIX)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(_QHPC_NVCC_FLAGS
    "-O3"
    "--use_fast_math"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda")
if(MSVC)
    # MSVC accepts forward-slash flags; commas separate inside -Xcompiler.
    # /wd4819 = silence "characters that cannot be represented in the current code page"
    # /wd4828 = silence "the file contains a character starting at offset 0x... that is illegal"
    # /wd4505 = silence "unreferenced local function has been removed" (MSVC equivalent of -Wno-unused-function)
    list(APPEND _QHPC_NVCC_FLAGS "-Xcompiler=/O2,/MD,/wd4819,/wd4828,/wd4505")
else()
    list(APPEND _QHPC_NVCC_FLAGS "-Xcompiler=-fPIC,-O3,-fno-strict-aliasing,-Wno-unused-function")
endif()
string(REPLACE ";" " " _QHPC_NVCC_FLAGS_STR "${_QHPC_NVCC_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_QHPC_NVCC_FLAGS_STR}")

# CUDAToolkit must be found BEFORE any target that links CUDA::* namespace.
find_package(CUDAToolkit REQUIRED)

# ----- The C++/CUDA shared object -----
add_library(qhpc_core SHARED
    qhpc_c_api.cu
)
set_target_properties(qhpc_core PROPERTIES
    # NOTE: must NOT be "qhpc" — that name is reserved for the Python module.
    # MSVC produces qhpc.lib for the .pyd module; if the core also called
    # itself qhpc.* there would be a LNK1149 file-name collision.
    OUTPUT_NAME      "qhpc_core"
    CUDA_SEPARABLE_COMPILATION ON
    # Auto-generate .lib import library on Windows from all extern "C" symbols.
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_include_directories(qhpc_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qhpc_core PUBLIC
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
    CUDA::nvrtc
    CUDA::cuda_driver)

if(QHPC_HAS_NVSHMEM)
    target_compile_definitions(qhpc_core PUBLIC QHPC_HAS_NVSHMEM=1)
    # User-provided nvshmem install path:
    target_link_libraries(qhpc_core PUBLIC nvshmem nvshmem_host)
endif()

# ----- Python module -----
# qhpc_pybind.cpp is HOST-ONLY C++ (no device code). Compiling it with NVCC
# pulls in pybind11's heavy template machinery through nvcc's C++ front-end,
# which is fragile on MSVC. We compile it as plain C++ and link against the
# already-built CUDA core (libqhpc_core).
if(QHPC_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(qhpc_py MODULE qhpc_pybind.cpp)
    set_target_properties(qhpc_py PROPERTIES
        OUTPUT_NAME "qhpc")
    target_include_directories(qhpc_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(qhpc_py PRIVATE qhpc_core)
endif()

# ----- Install -----
install(TARGETS qhpc_core
        LIBRARY DESTINATION lib)
install(FILES qhpc_c_api.h DESTINATION include)
if(QHPC_BUILD_PYTHON)
    install(TARGETS qhpc_py
            LIBRARY DESTINATION lib/python)
endif()

message(STATUS "QHPC: SM arch       = ${QHPC_SM_ARCH}")
message(STATUS "QHPC: Python module = ${QHPC_BUILD_PYTHON}")
message(STATUS "QHPC: NVSHMEM       = ${QHPC_HAS_NVSHMEM}")
