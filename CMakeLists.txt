cmake_minimum_required(VERSION 3.15)
project(openbw)

set(CMAKE_CXX_STANDARD 14)

include(ExternalProject)

ExternalProject_Add(
    Starcraft_1161
    PREFIX "StarCraft"
    URL http://www.cs.mun.ca/~dchurchill/starcraftaicomp/files/Starcraft_1161.zip
    URL_HASH SHA512=8bc0f8f576bccac5c4f5ddf359b51c22bc2b9f2fd9213164b1c294fd2e7538e085c12868731905b8108f6e09e8d9c08fab271f5426ec0220774dd3240224917d
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE TRUE
    BUILD_COMMAND ${CMAKE_COMMAND} -E copy BROODAT.MPQ STARDAT.MPQ patch_rt.mpq ${PROJECT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory maps ${PROJECT_BINARY_DIR}/maps
    BUILD_BYPRODUCTS ${PROJECT_BINARY_DIR}/BROODAT.MPQ ${PROJECT_BINARY_DIR}/STARDAT.MPQ ${PROJECT_BINARY_DIR}/patch_rt.mpq ${PROJECT_BINARY_DIR}/maps
    INSTALL_COMMAND ""
)

# pybind11 python library.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)
pybind11_add_module(bwgame pybind.cc)
target_include_directories(bwgame PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openbw)
