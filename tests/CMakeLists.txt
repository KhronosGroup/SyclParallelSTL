

add_definitions(-std=c++11)

set(GMOCK_DIR "${CMAKE_SOURCE_DIR}/external/gmock/")

if (MSVC)
	set(GMOCK_LIB_PATH ${GMOCK_DIR}/msvc/2010/x64/Debug/gmock.lib)
	set(GMOCK_MAIN_LIB_PATH ${GMOCK_DIR}/msvc/2010/x64/Debug/gmock_main.lib)
else()
	set(GMOCK_LIB_PATH ${GMOCK_DIR}/make/gmock.a)
	set(GMOCK_MAIN_LIB_PATH ${GMOCK_DIR}/make/gmock_main.a)
endif()

if (EXISTS ${GMOCK_LIB_PATH})
  message(STATUS " GoogleMock found in ${GMOCK_DIR} , building tests ")
  include_directories(${GMOCK_DIR}/include)
  include_directories(${GMOCK_DIR}/gtest/include)

  file(GLOB SRC_FILES "*.cpp")

  if (SRC_FILES)
    add_library(gmock STATIC IMPORTED)
	add_library(gmock_main STATIC IMPORTED)
    set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${GMOCK_LIB_PATH})
	set_property(TARGET gmock_main PROPERTY IMPORTED_LOCATION ${GMOCK_MAIN_LIB_PATH})
	
    foreach(file ${SRC_FILES})
      get_filename_component(filename ${file} NAME)
      set(exe_name "${filename}.exe")
      message(STATUS "File ${file} Filename ${filename} Exename ${exe_name} ")
      add_executable (${exe_name} ${file})
	  if(MSVC)
		target_link_libraries(${exe_name} gmock gmock_main SyclSTL)
	  else()
		target_link_libraries(${exe_name} gmock gmock_main SyclSTL pthread)
	  endif()

      set_property(TARGET ${exe_name} PROPERTY CXX_STANDARD "11")
      set_property(TARGET ${exe_name} PROPERTY CXX_STANDARD_REQUIRED True)

      if (NOT HEADER_ONLY)
        # From the SYCL package
        # TODO(Ruyman): build_spir should return the output file name
        build_spir(${exe_name} ${file} ${CMAKE_CURRENT_BINARY_DIR})
		if (MSVC)
			set_property(TARGET ${exe_name} PROPERTY COMPILE_FLAGS 
				"-std=c++11 /FI${CMAKE_CURRENT_BINARY_DIR}/${filename}.sycl ${CMAKE_CXX_FLAGS}")
		else()
			set_property(TARGET ${exe_name} PROPERTY COMPILE_FLAGS 
				"-std=c++11 -include ${CMAKE_CURRENT_BINARY_DIR}/${filename}.sycl ${CMAKE_CXX_FLAGS}")
		endif()
        add_dependencies(${exe_name} ${file}.sycl)
      endif ()
    endforeach()

  else()
    message(WARNING "No tests found")
  endif()


else()
  message(WARNING " Tests are not enabled, GoogleMock not found ")
  message(WARNING " Note that you need to build the gmock_main file in ${GMOCK_DIR}/make")
endif()
