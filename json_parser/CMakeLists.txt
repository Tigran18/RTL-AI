cmake_minimum_required(VERSION 3.14...3.28)
project(json_parser CXX)

add_library(json_parser STATIC
    json_parser.cpp
)
target_include_directories(json_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})