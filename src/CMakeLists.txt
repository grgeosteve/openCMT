option(BUILD_CMT_SHARED "Build CMT shared libraries." ON)

if(BUILD_CMT_SHARED)
    set(CMT_LIBRARY_TYPE SHARED)
else(BUILD_CMT_SHARED)
    set(CMT_LIBRARY_TYPE STATIC)
endif(BUILD_CMT_SHARED)

add_library(${PROJECT_NAME} ${CMT_LIBRARY_TYPE} utils.cpp cluster.cpp cmt.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
