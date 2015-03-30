project(openCMT)
cmake_minimum_required(VERSION 2.8)

set(${CMAKE_BUILD_TYPE} Release)

find_package(OpenCV REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(src)

