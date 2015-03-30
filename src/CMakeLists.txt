option(BUILD_CMT_SHARED "Build CMT shared libraries." ON)

if(BUILD_CMT_SHARED)
	set(CMT_LIBRARY_TYPE SHARED)
else(BUILD_CMT_SHARED)
	set(CMT_LIBRARY_TYPE STATIC)
endif(BUILD_CMT_SHARED)

add_library(${PROJECT_NAME} ${CMT_LIBRARY_TYPE} utils.cpp cluster.cpp cmt.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

option(BUILD_MAIN "Build main(CMT_Tracker executable)" ON)
if (BUILD_MAIN)
   add_executable(CMT_Tracker parser.cpp main.cpp)
   target_link_libraries(CMT_Tracker ${OpenCV_LIBS} ${PROJECT_NAME})
endif(BUILD_MAIN)
