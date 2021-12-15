cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(moolib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -ftemplate-backtrace-limit=0 -Bsymbolic")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

execute_process(
    COMMAND python -c "import os, torch; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

message(STATUS "PyTorch compilation flags: ${TORCH_CXX_FLAGS}")
message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Threads REQUIRED)

option(USE_CUDA "Enable CUDA support" ON)

if (USE_CUDA)
  option(TP_USE_CUDA "" ON)
endif()

add_subdirectory(third_party/fmt)
add_subdirectory(src/tensorpipe)
add_subdirectory(third_party/pybind11)

#add_subdirectory(test)

add_library(tensor src/tensor.cc)
target_link_libraries(tensor PRIVATE ${TORCH_LIBRARIES})
target_include_directories(tensor SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_options(tensor PRIVATE ${TORCH_CXX_FLAGS})

add_library(moorpc src/rpc.cc src/async.cc)
target_link_libraries(moorpc PUBLIC tensor fmt tensorpipe)
target_include_directories(moorpc PRIVATE src/tensorpipe src)

if (USE_CUDA)
  target_compile_definitions(tensor PRIVATE -DUSE_CUDA)
  target_compile_definitions(moorpc PRIVATE -DUSE_CUDA)
endif()

pybind11_add_module(
  _C
  src/moolib.cc
  src/accumulator.cc
  src/batch_utils.cc
  src/env.cc
  src/tensorpython.cc
)
target_link_libraries(_C PRIVATE moorpc ${TorchPath}/lib/libtorch_python${CMAKE_SHARED_LIBRARY_SUFFIX})

set_source_files_properties(src/tensorpython.cc PROPERTIES INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}")
set_source_files_properties(src/tensorpython.cc PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}")

if (USE_BACKWARD)
  find_library(DW_LIB dw REQUIRED)
  find_library(UNWIND_LIB unwind REQUIRED)
  find_path(DW_INCLUDE dwarf.h REQUIRED)

  target_sources(_C PUBLIC third_party/backward-cpp/backward.cpp)
  target_include_directories(_C PRIVATE ${DW_INCLUDE})
  target_link_libraries(_C PUBLIC moorpc ${DW_LIB} ${UNWIND_LIB})
  target_compile_definitions(_C PRIVATE BACKWARD_HAS_DW BACKWARD_HAS_LIBUNWIND)
endif()

