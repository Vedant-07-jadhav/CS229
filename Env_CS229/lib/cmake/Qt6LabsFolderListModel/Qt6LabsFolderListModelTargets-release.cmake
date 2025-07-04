#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Qt6::LabsFolderListModel" for configuration "Release"
set_property(TARGET Qt6::LabsFolderListModel APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qt6::LabsFolderListModel PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "Qt6::Qml"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libQt6LabsFolderListModel.so.6.7.3"
  IMPORTED_SONAME_RELEASE "libQt6LabsFolderListModel.so.6"
  )

list(APPEND _cmake_import_check_targets Qt6::LabsFolderListModel )
list(APPEND _cmake_import_check_files_for_Qt6::LabsFolderListModel "${_IMPORT_PREFIX}/lib/libQt6LabsFolderListModel.so.6.7.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
