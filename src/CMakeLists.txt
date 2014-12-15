add_library(${PROJECT_NAME} tools.cpp cluster.cpp cmt.cpp)
target_link_libraries(CMT ${OpenCV_LIBS})
