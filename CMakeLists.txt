cmake_minimum_required(VERSION 3.26)
project(Tracking)
add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)
set(OpenCV_DIR "C:/tools/opencv/build")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 23)

add_executable(Tracking
        src/main.cpp
        include/Gradient.h
        include/Line.h
        include/HOG.h
        src/train_svm.cpp
        include/Detection.h
)


target_link_libraries(Tracking LINK_PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
