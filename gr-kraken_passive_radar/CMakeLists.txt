cmake_minimum_required(VERSION 3.8)
project(gr-kraken_passive_radar CXX C)

set(CMAKE_CXX_STANDARD 17)

find_package(Gnuradio "3.10" REQUIRED)

include(GrVersion)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib
    ${GNURADIO_ALL_INCLUDE_DIRS}
)

set(GR_REQUIRED_COMPONENTS RUNTIME)
find_package(Gnuradio "3.10" REQUIRED COMPONENTS ${GR_REQUIRED_COMPONENTS})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(include/gnuradio/kraken_passive_radar)
add_subdirectory(lib)
add_subdirectory(apps)
add_subdirectory(docs)
add_subdirectory(python/kraken_passive_radar)
add_subdirectory(grc)

# Install cmake modules
install(FILES cmake/Modules/kraken_passive_radarConfig.cmake
    DESTINATION lib/cmake/kraken_passive_radar
)
