//
// Created by janra on 26.11.2023.
//
#include "../libs/Eigen/Dense"
#ifndef TRACKING_HOG_H
#define TRACKING_HOG_H

cv::Mat hog(const Eigen::MatrixXd &image,
         int orientations = 9,
         std::pair<int, int> pixels_per_cell = std::make_pair(8, 8),
         std::pair<int, int> cells_per_block = std::make_pair(2, 2),
         std::string block_norm = "L2-Hys",
         bool visualize = false,
         bool transform_sqrt = true,
         bool feature_vector = true);

#endif //TRACKING_HOG_H
