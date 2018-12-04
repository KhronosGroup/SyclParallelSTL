

set(files 
  "all_of.cpp"
  "any_of.cpp"
  "buffer_unique_ptr.cpp"
  "count.cpp"
  "count_if.cpp"
  "equal.cpp"
  "exclusive_scan.cpp"
  "fill.cpp"
  "fill_n.cpp"
  "find.cpp"
  "for_each.cpp"
  "for_each_n.cpp"
  "generate.cpp"
  "generate_n.cpp"
  "inclusive_scan.cpp"
  "inner_product.cpp"
  "invalid_vector_iterators.cpp"
  "mismatch.cpp"
  "none_of.cpp"
  "reduce.cpp"
  "replace_copy.cpp"
  "replace_copy_if.cpp"
  "replace.cpp"
  "replace_if.cpp"
  "reverse_copy.cpp"
  "reverse.cpp"
  "sort.cpp"
  "transform.cpp"
  "transform_reduce.cpp"
  "vector.cpp"
  )

foreach (file ${files})
    build_file(${file})
    get_filename_component(SOURCE_NAME ${file} NAME_WE)
    target_link_libraries(${SOURCE_NAME} PUBLIC gtest_main)
    add_test(${SOURCE_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_NAME})
endforeach()
