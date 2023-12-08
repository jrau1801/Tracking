//
// Created by Louis-Kaan Ay on 08.12.23.
//

#ifndef TRACKING_SLIDINGWINDOW_H
#define TRACKING_SLIDINGWINDOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Represents functionality for generating sliding windows.
 */
class SlidingWindow {

public:
    SlidingWindow() = default;

    SlidingWindow(const SlidingWindow &) = delete;

    SlidingWindow(SlidingWindow &&) = delete;

    SlidingWindow &operator=(const SlidingWindow &) = delete;

    SlidingWindow &operator=(SlidingWindow &&) = delete;

    ~SlidingWindow() = default;

    /**
     * @brief Generates sliding windows over the given image.
     *
     * @param image The input image.
     * @param windowSize Size of the sliding window.
     * @param stepSize Step size for sliding.
     * @return Vector containing generated sliding windows.
     */
    static std::vector<cv::Rect> generate(const cv::Mat &image, cv::Size windowSize, cv::Size stepSize) {
        std::vector<cv::Rect> windows;
        for (int y = 0; y < image.rows - windowSize.height; y += stepSize.height) {
            for (int x = 0; x < image.cols - windowSize.width; x += stepSize.width) {
                windows.push_back(cv::Rect(x, y, windowSize.width, windowSize.height));
            }
        }
        return windows;
    }

};


#endif //TRACKING_SLIDINGWINDOW_H
