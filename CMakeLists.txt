cmake_minimum_required(VERSION 3.26)
project(Tracking)

set(CMAKE_CXX_STANDARD 23)


add_executable(Tracking src/draw_new.cpp
        src/hoghistogram.cpp
        src/hog.cpp
        include/hoghistogram.h
        src/main.cpp
        include/draw_new.h
)
