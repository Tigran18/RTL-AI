add_library(json_parser STATIC json_parser.cpp)
target_compile_options(json_parser PRIVATE -fPIC)
target_include_directories(json_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
