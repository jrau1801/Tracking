cmake_minimum_required(VERSION 3.26)
project(Tracking)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 23)


add_executable(Tracking
        src/draw_new.cpp
        src/hoghistogram.cpp
        src/hog.cpp
        src/main.cpp
        include/draw_new.h
        include/hoghistogram.h
)


target_link_libraries(Tracking ${OpenCV_LIBS})
