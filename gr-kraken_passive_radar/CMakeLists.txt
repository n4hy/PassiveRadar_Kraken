cmake_minimum_required(VERSION 3.8)
project(gr-kraken_passive_radar CXX C)

set(CMAKE_CXX_STANDARD 17)

find_package(Gnuradio "3.10" REQUIRED COMPONENTS RUNTIME)

include(GrVersion)
include(GrPython)
# DO NOT include(GrModule) on this platform

# Python blocks
add_subdirectory(python/kraken_passive_radar)

# GRC block definitions
add_subdirectory(grc)

# C++ ECA-B kernel library (build from top-level src/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/src_build)

