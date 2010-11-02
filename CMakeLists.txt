cmake_minimum_required(VERSION 2.6)
project(tinyest)

include (CheckIncludeFile)

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

include_directories(${tinyest_SOURCE_DIR})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -pedantic")

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
endif()

# SSE/SSE2
set(CMAKE_TINYEST_HAS_SSE 0)
set(ENABLE_SSE2 OFF CACHE BOOL "Compilation with SSE2/MMX extensions")
if(ENABLE_SSE2)
  set(CMAKE_TINYEST_HAS_SSE 1)
endif()

# OpenMP
#find_package(OpenMP)
#if(OPENMP_FOUND)
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#endif()

if (CMAKE_TINYEST_HAS_SSE)
  CHECK_INCLUDE_FILE("emmintrin.h" HAVE_EMMINTRIN_H)

  if (HAVE_EMMINTRIN_H)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_SSE -DHAVE_EMMINTRIN_H")
  endif()
endif()

find_library(M_LIB m)

set (LIBTINYEST_SOURCES
  src/dataset.c
  src/lbfgs.c
  src/maxent.c
  src/model.c
)

add_library(tinyest SHARED
  ${LIBTINYEST_SOURCES}
)

target_link_libraries(tinyest ${ZLIB_LIBRARIES})

add_executable(estimate
  util/estimate.c
)

target_link_libraries(estimate
  tinyest
  ${M_LIB}
)
