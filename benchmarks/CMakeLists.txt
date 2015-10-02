file(GLOB EXAMPLE_FILES ./*.cpp)

foreach(file ${EXAMPLE_FILES})
  get_filename_component(filename ${file} NAME)
  set(exe_name "${filename}.exe")

  message(STATUS "File ${file} Filename ${filename}  Exename ${exe_name}")
  add_sycl_target(${exe_name} ${file} ${CMAKE_CURRENT_BINARY_DIR})

endforeach()
