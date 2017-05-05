set(GMOCK_DIR "${CMAKE_SOURCE_DIR}/external/googletest/googlemock/")

if (MSVC)
	set(GMOCK_LIB_PATH ${GMOCK_DIR}/msvc/2010/x64/Debug/gmock.lib)
	set(GMOCK_MAIN_LIB_PATH ${GMOCK_DIR}/msvc/2010/x64/Debug/gmock_main.lib)
else()
	set(GMOCK_LIB_PATH ${GMOCK_DIR}/make/gmock_main.a)
	set(GMOCK_MAIN_LIB_PATH ${GMOCK_DIR}/make/gmock_main.a)
endif()

if (EXISTS ${GMOCK_LIB_PATH})
  message(STATUS " GoogleMock found in ${GMOCK_DIR} , building tests ")
  include_directories(${GMOCK_DIR}/include)
  include_directories(${GMOCK_DIR}/gtest/include)
  include_directories(${GMOCK_DIR}/../googletest/include)

  file(GLOB SRC_FILES "*.cpp")

  if (SRC_FILES)
    add_library(gmock STATIC IMPORTED)
	  add_library(gmock_main STATIC IMPORTED)
    set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${GMOCK_LIB_PATH})
	  set_property(TARGET gmock_main PROPERTY IMPORTED_LOCATION ${GMOCK_MAIN_LIB_PATH})
	
    foreach(file ${SRC_FILES})
            get_filename_component(SOURCE_NAME ${file} NAME_WE)
            set(TEST_NAME "test_${SOURCE_NAME}")
            message(STATUS " Adding ${SOURCE_NAME} ")

            include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

            add_executable(${TEST_NAME} 
                      ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp  )
            target_compile_options(${TEST_NAME} PUBLIC ${HOST_COMPILER_OPTIONS})
            target_link_libraries(${TEST_NAME} PUBLIC ${GMOCK_LIB_PATH})
            target_link_libraries(${TEST_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

            add_sycl_to_target(${TEST_NAME} 
                      ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp 
                      ${CMAKE_CURRENT_BINARY_DIR})

            add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

  else()
    message(WARNING "No tests found")
  endif()

else()
  message(WARNING " Tests are not enabled, GoogleMock not found ")
  message(WARNING " Note that you need to build the gmock_main file in ${GMOCK_DIR}/make")
endif()
