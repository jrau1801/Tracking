//
// Created by Louis-Kaan Ay on 28.11.23.
//

#ifndef TRACKING_GRADIENT_H
#define TRACKING_GRADIENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using cv::Mat;


class Gradient {
public:
    static std::pair<Mat, Mat> compute_gradients_sobel(const Mat &input) {
        Mat grad_x, grad_y;

        // Compute gradients using Sobel in the x and y directions
        cv::Sobel(input, grad_x, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::Sobel(input, grad_y, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

        return std::make_pair(grad_y, grad_x);
    }


    static std::pair<Mat, Mat> compute_gradients_prewitt(const Mat &input) {
        Mat grad_x, grad_y;

        // Compute gradients using Prewitt operator
        Mat kernel_x = (cv::Mat_<double>(3, 3) << -1, 0, 1,
                -1, 0, 1,
                -1, 0, 1);

        Mat kernel_y = (cv::Mat_<double>(3, 3) << -1, -1, -1,
                0, 0, 0,
                1, 1, 1);

        cv::filter2D(input, grad_x, -1, kernel_x);
        cv::filter2D(input, grad_y, -1, kernel_y);

        return std::make_pair(grad_y, grad_x);
    }

    Gradient() = delete;



};


#endif //TRACKING_GRADIENT_H
