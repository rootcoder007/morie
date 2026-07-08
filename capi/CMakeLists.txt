# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Standalone build for libmorie_core -- the C ABI face of the morie numeric
# core. Deliberately SEPARATE from the repo-root CMakeLists.txt (which is the
# scikit-build-core / nanobind wheel build) so building the C library never
# perturbs the Python wheel. Build it on its own:
#
#     cmake -S capi -B capi/build
#     cmake --build capi/build
#     ctest --test-dir capi/build --output-on-failure
#
# Produces libmorie_core (shared + static) + the public header, installable,
# and a pure-C test that links the shared library.

cmake_minimum_required(VERSION 3.15)
project(libmorie_core LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The façade translation unit (C++, wraps the header-only core).
add_library(morie_core_obj OBJECT morie_c_api.cpp)
target_include_directories(morie_core_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})

# Shared + static from the same objects.
add_library(morie_core SHARED $<TARGET_OBJECTS:morie_core_obj>)
add_library(morie_core_static STATIC $<TARGET_OBJECTS:morie_core_obj>)
foreach(t morie_core morie_core_static)
  target_include_directories(${t} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
set_target_properties(morie_core_static PROPERTIES OUTPUT_NAME morie_core)

# Pure-C consumer, linked against the shared lib -> proves the C ABI.
enable_testing()
add_executable(test_c_api test_c_api.c)
target_link_libraries(test_c_api PRIVATE morie_core)
if(NOT WIN32)
  target_link_libraries(test_c_api PRIVATE m) # libm for math.h
endif()
add_test(NAME c_abi COMMAND test_c_api)

install(TARGETS morie_core morie_core_static
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES morie_c_api.h DESTINATION include)
