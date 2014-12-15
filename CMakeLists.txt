cmake_minimum_required(VERSION 2.6)

project(CMT)
set(CMT_VERSION_MAJOR 1)
set(CMT_VERSION_MINOR 0)

find_package(OpenCV REQUIRED)

add_subdirectory(src)
