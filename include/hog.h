//
// Created by janra on 26.11.2023.
//
#include "../libs/Eigen/Dense"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#ifndef TRACKING_HOG_H
#define TRACKING_HOG_H

std::pair<cv::Mat, cv::Mat> hog(const cv::Mat &image,
             int orientations = 9,
            std::pair<int, int> pixels_per_cell = std::make_pair(8, 8),
            std::pair<int, int> cells_per_block = std::make_pair(3, 3),
             std::string method = "L2-Hys",
            bool visualize = false,
             bool transform_sqrt = true,
             bool flatten = true);

#endif //TRACKING_HOG_H
