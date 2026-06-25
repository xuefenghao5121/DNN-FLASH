# Find a system or project-local oneDNN/DNNL installation.
# Exports:
#   LocalDnnl_FOUND / DNNL_FOUND
#   DNNL_INCLUDE_DIR
#   DNNL_LIBRARY
#   dnnl::dnnl imported target

set(_FLASHONE_DNNL_HINTS
    "${CMAKE_SOURCE_DIR}/third_party/onednn-local/usr"
    "/usr"
    "/usr/local")

find_path(DNNL_INCLUDE_DIR
    NAMES dnnl.hpp dnnl.h
    HINTS ${_FLASHONE_DNNL_HINTS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)

find_library(DNNL_LIBRARY
    NAMES dnnl
    HINTS ${_FLASHONE_DNNL_HINTS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)

# Fall back to normal system search after explicit hints.
if(NOT DNNL_INCLUDE_DIR)
    find_path(DNNL_INCLUDE_DIR NAMES dnnl.hpp dnnl.h)
endif()
if(NOT DNNL_LIBRARY)
    find_library(DNNL_LIBRARY NAMES dnnl)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LocalDnnl DEFAULT_MSG DNNL_INCLUDE_DIR DNNL_LIBRARY)

set(DNNL_FOUND ${LocalDnnl_FOUND})

if(LocalDnnl_FOUND AND NOT TARGET dnnl::dnnl)
    add_library(dnnl::dnnl UNKNOWN IMPORTED)
    set_target_properties(dnnl::dnnl PROPERTIES
        IMPORTED_LOCATION "${DNNL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${DNNL_INCLUDE_DIR}")
endif()
