cmake_minimum_required(VERSION 2.6)
project(tinyest)

include (CheckIncludeFile)

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

include_directories(${tinyest_SOURCE_DIR}/include)

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

set (LIBTINYEST_HEADERS
  include/tinyest/bitvector.h
  include/tinyest/dataset.h
  include/tinyest/lbfgs.h
  include/tinyest/maxent.h
  include/tinyest/model.h
)

set (LIBTINYEST_SOURCES
  src/bitvector.c
  src/dataset.c
  src/lbfgs.c
  src/maxent.c
  src/model.c
)

add_library(tinyest SHARED
  ${LIBTINYEST_SOURCES}
  ${LIBTINYEST_HEADERS}
)

target_link_libraries(tinyest ${ZLIB_LIBRARIES})

add_executable(tinyest_executable
  util/estimate.c
)

set_target_properties(tinyest_executable PROPERTIES OUTPUT_NAME "tinyest")

target_link_libraries(tinyest_executable
  tinyest
  ${M_LIB}
)

install(FILES ${LIBTINYEST_HEADERS} DESTINATION include/tinyest)
install(TARGETS tinyest_executable tinyest 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)
