#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DNNL::dnnl" for configuration "RelWithDebInfo"
set_property(TARGET DNNL::dnnl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(DNNL::dnnl PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libdnnl.so.3.1"
  IMPORTED_SONAME_RELWITHDEBINFO "libdnnl.so.3"
  )

list(APPEND _cmake_import_check_targets DNNL::dnnl )
list(APPEND _cmake_import_check_files_for_DNNL::dnnl "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libdnnl.so.3.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
