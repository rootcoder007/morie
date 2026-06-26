# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build for libmorie -- the morie C++ numeric core (v0.9.1 backend
# port). Driven by scikit-build-core; produces the nanobind extension
# module installed as ``morie._core``.

cmake_minimum_required(VERSION 3.15...3.31)
project(morie_core LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Emit compile_commands.json so the clangd language server (see the
# repo-root .clangd) can resolve the nanobind / Python build headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# scikit-build-core points CMake at the correct Python; nanobind is
# installed into the build-isolation environment via build-system
# requires, so its CMake package config is discoverable.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NB_STATIC links nanobind's runtime statically -- one self-contained
# extension, no separate shared nanobind dependency in the wheel.
nanobind_add_module(_core NB_STATIC
  libmorie/_core.cpp
  libmorie/kernels.cpp
  libmorie/hawkes.cpp)

# Land the compiled module inside the morie package: morie/_core*.so,
# importable as ``morie._core``.
install(TARGETS _core LIBRARY DESTINATION morie)

# --- Bundle morie.fn (keep the wheel from shipping ~73k loose files) ---------
# scripts/bundle_fn.py collapses the ~36k per-callable .py into _fnsrc.json.xz
# (imported via zipimport on morie.fn.__path__) and the ~36k describe_*.md into
# describe_docs.json.xz (read by morie.fn.describe()). The loose files are
# dropped from the wheel via wheel.exclude in pyproject.toml; these two archives
# are installed into morie/fn/ in their place. The loaders fall back to loose
# files, so the sdist / dev tree (which keep them) still work unchanged.
set(_FN_BUNDLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fn_bundle")
add_custom_command(
  OUTPUT "${_FN_BUNDLE_DIR}/_fnsrc.json.xz" "${_FN_BUNDLE_DIR}/describe_docs.json.xz"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${_FN_BUNDLE_DIR}"
  COMMAND "${Python_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_fn.py"
          --src "${CMAKE_CURRENT_SOURCE_DIR}/src/morie/fn"
          --out "${_FN_BUNDLE_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_fn.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/morie/fn/_lazy_map.json"
  COMMENT "Bundling morie.fn -> _fnsrc.json.xz + describe_docs.json.xz"
  VERBATIM)
add_custom_target(fn_bundle ALL
  DEPENDS "${_FN_BUNDLE_DIR}/_fnsrc.json.xz" "${_FN_BUNDLE_DIR}/describe_docs.json.xz")
install(FILES "${_FN_BUNDLE_DIR}/_fnsrc.json.xz" "${_FN_BUNDLE_DIR}/describe_docs.json.xz"
        DESTINATION morie/fn)
