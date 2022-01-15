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
pybind11_add_module(bwgame pybind.cc pybwenums.cc)
target_include_directories(bwgame PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openbw)

find_package(SDL2 REQUIRED)

find_path(SDL2_IMAGE_INCLUDE_DIR
  SDL_image.h
  PATH_SUFFIXES SDL2
  include/SDL2 include
)
find_library(SDL2_IMAGE_LIBRARY
  NAMES SDL2_image
  PATH_SUFFIXES lib
)

find_path(SDL2_MIXER_INCLUDE_DIR
  SDL_mixer.h
  PATH_SUFFIXES SDL2
  include/SDL2 include
)
find_library(SDL2_MIXER_LIBRARY
  NAMES SDL2_mixer
  PATH_SUFFIXES lib
)

add_executable(gfxtest
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openbw/ui/gfxtest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openbw/ui/sdl2.cpp
  cc/mallocfree.c
)
target_include_directories(gfxtest
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openbw
  ${SDL2_INCLUDE_DIRS}
  ${SDL2_IMAGE_INCLUDE_DIR}
)
target_link_libraries(gfxtest
  ${SDL2_LIBRARIES}
  ${SDL2_IMAGE_LIBRARY}
  ${SDL2_MIXER_LIBRARY}
)

add_custom_command(
  DEPENDS "flash_vs_jaedong.rep"
  OUTPUT maps/p49.rep
  COMMAND ${CMAKE_COMMAND} -E create_symlink "../../flash_vs_jaedong.rep" maps/p49.rep
)
add_custom_target(link_replay ALL DEPENDS maps/p49.rep)
